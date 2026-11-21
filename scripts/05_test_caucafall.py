import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import CONFIG_PATH, test_frozen  # noqa: E402


def print_inputs(run):
    windows, truth = run["windows"], run["truth"]
    print(f"test datasets: {run['datasets']}")
    print(f"datasets the frozen detectors were trained on that are also tested here: {run['trained_on_tested'] or 'none'}")
    for detector in run["detectors"]:
        print(f"  {detector['name']}: threshold {detector['threshold']:.6g}, trained on {detector['train_datasets']}, params {detector['params']}")
    print(f"labelled clips: {len(truth)} (fall {int(truth.sum())}, non fall {int((truth == 0).sum())})")
    print(f"clips with windows: {windows['clip_id'].nunique()}")
    print("windows per dataset and class:")
    print(windows.groupby(["dataset", "label"]).size().to_string())
    windowless = run["windowless"]
    print(f"clips without windows, counted as predicted not fall and kept in clip level denominators: {len(windowless)}")
    for clip_id, label in windowless.items():
        print(f"  {clip_id}: true label {label}")
    positive_clips = set(windows.loc[windows["label"] == 1, "clip_id"])
    unwindowed_falls = sorted(set(truth[truth == 1].index) & set(windows["clip_id"]) - positive_clips)
    print(f"fall clips with windows but no positive window: {unwindowed_falls or 'none'}")


def print_results(run):
    outputs = run["outputs"]
    results = pd.read_csv(outputs["results"])
    columns = ["detector", "variant", "level", "threshold", "recall", "precision", "specificity", "f1", "accuracy",
               "n_test", "n_pos", "tp", "fp", "tn", "fn"]
    print(f"metrics, read back from {outputs['results']}:")
    print(results[columns].round(3).to_string(index=False))
    print(f"window predictions: {outputs['window_predictions']}")
    print(f"clip predictions: {outputs['clip_predictions']}")
    print(f"exact McNemar on clip level predictions, read back from {outputs['mcnemar']}:")
    print(pd.read_csv(outputs["mcnemar"]).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Score the frozen detectors once on the unseen test set.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run = test_frozen(args.config, force=args.force)
    print_inputs(run)
    print_results(run)


if __name__ == "__main__":
    main()
