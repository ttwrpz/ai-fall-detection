import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import CONFIG_PATH, cross_validate  # noqa: E402


def print_inputs(run):
    windows, truth = run["windows"], run["clip_truth"]
    windowed = truth.loc[windows["clip_id"].unique()]
    print("windows per dataset and class:")
    print(windows.groupby(["dataset", "label"]).size().to_string())
    print(f"features: {len(run['features'])}")
    print(f"clips with windows: {len(windowed)} (fall {int(windowed.sum())}, non fall {int((windowed == 0).sum())})")
    windowless = pd.read_csv(run["windowless"])
    print(f"labelled training clips without windows, excluded from clip level scores, read back from {run['windowless']}:")
    print(windowless.to_string(index=False))
    print(
        f"total without windows: fall {windowless['fall_clips_without_windows'].sum()}, "
        f"non fall {windowless['non_fall_clips_without_windows'].sum()}"
    )
    print("folds:")
    print(run["folds"].to_string(index=False))


def print_results(run):
    results = pd.read_csv(run["cv_results"])
    means = results.groupby(["level", "detector", "variant"])[["recall", "f1", "threshold"]].mean()
    print(f"mean over folds, read back from {run['cv_results']}:")
    print(means.round(3).to_string())
    print(f"rows written: {len(results)} to {run['cv_results']}")
    print(f"summary written: {run['tuning']}")
    oof = pd.read_csv(run["oof"])
    print(f"out of fold window predictions written: {len(oof)} rows to {run['oof']}")
    print(oof.groupby(["detector", "variant"]).agg(windows=("clip_id", "size"), folds=("fold", "nunique")).to_string())


def print_selection(frozen):
    selection = pd.read_csv(frozen["selection"])
    chosen = selection[selection["chosen"]].iloc[0]
    print(f"model selection, read back from {frozen['selection']}:")
    print(selection.round(3).to_string(index=False))
    print(f"chosen: {chosen['name']} (mean CV window F1 {chosen['f1_mean']:.3f}, recall {chosen['recall_mean']:.3f})")
    if not chosen["reaches_recall_target"]:
        print(
            f"WARNING: no candidate reaches mean CV window recall {frozen['recall_target']}, "
            f"fell back to the highest mean F1: {chosen['name']} with recall {chosen['recall_mean']:.3f}"
        )


def print_frozen(run):
    frozen, windows = run["frozen"], run["windows"]
    best, rule = frozen["best"], frozen["rule"]
    print("refit on all training windows, per dataset and class:")
    print(windows.groupby(["dataset", "label"]).size().to_string())
    print(f"training clips: {windows['clip_id'].nunique()}, windows: {len(windows)}")
    print(f"test dataset rows used in refit: {frozen['test_rows_used']}")
    print(f"non training rows in parquet, excluded: {frozen['excluded_rows'] or 0}")
    print(f"saved {frozen['best_path']}: name {best['name']}, threshold {best['threshold']:.6g}, params {best['params']}")
    print(f"  keys: {sorted(best)}, score {best['score']} via {best['score_method']}, features {len(best['features'])}")
    print(f"saved {frozen['rule_path']}: name {rule['name']}, threshold {rule['threshold']}")
    print(f"  fitted rule thresholds: {rule['params']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    start = time.perf_counter()
    run = cross_validate(args.config)
    print_inputs(run)
    print_results(run)
    print_selection(run["frozen"])
    print_frozen(run)
    print(f"runtime: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
