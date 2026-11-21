import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.report import CONFIG_PATH, build_report  # noqa: E402


def print_inputs(run):
    oof = run["oof"]
    print("out of fold window predictions per dataset and class (one detector):")
    print(oof[oof["name"] == run["models"][0]].groupby(["dataset", "label"]).size().to_string())
    print(f"folds: {sorted(oof['fold'].unique().tolist())}, detectors: {sorted(oof['name'].unique())}")
    print(f"selected variant per model: {run['models']}")
    clips = run["clips"]
    print("non fall training clips by activity source:")
    print(clips.groupby(["dataset", "activity_source"]).agg(clips=("clip_id", "size"), activities=("activity", "nunique")).to_string())


def print_outputs(run):
    outputs = run["outputs"]
    print(f"pooled outer fold metrics, read back from {outputs['pooled']}:")
    print(pd.read_csv(outputs["pooled"]).round(3).to_string(index=False))
    importance = pd.read_csv(outputs["importance"])
    print(f"top {run['top_n']} features, read back from {outputs['importance']}:")
    print(importance.head(run["top_n"])[["rank", "feature", "importance"]].round(4).to_string(index=False))
    alarms = pd.read_csv(outputs["alarms"])
    rates = [column for column in alarms.columns if column.endswith("_rate") or column.endswith("_alarmed")]
    totals = alarms[alarms["activity_source"] == "dataset total"]
    print(f"activity alarm rates, {len(alarms)} rows, dataset totals read back from {outputs['alarms']}:")
    print(totals[["dataset", "clips", "windows", *rates]].round(3).to_string(index=False))
    flagged = alarms[(alarms["activity_source"] != "dataset total") & (alarms[rates].filter(like="clips_alarmed").sum(axis=1) > 0)]
    print(f"activities where any of {run['alarm_names']} fired at least once: {len(flagged)}")
    print(flagged[["dataset", "activity", "clips", *rates]].round(3).to_string(index=False))
    print(f"dataset summary, read back from {outputs['datasets']}:")
    print(pd.read_csv(outputs["datasets"]).to_string(index=False))
    lstm = pd.read_csv(outputs["lstm"])
    columns = ["detector", "variant", "level", "folds"] + [c for c in lstm.columns if c.startswith(("recall_", "f1_"))]
    print(f"LSTM over folds, read back from {outputs['lstm']}:")
    print(lstm[columns].round(3).to_string(index=False))
    print(f"detection delay from outer fold windows (first positive window end), read back from {outputs['delay_summary']}:")
    print(pd.read_csv(outputs["delay_summary"]).round(3).to_string(index=False))
    tradeoff = pd.read_csv(outputs["tradeoff"])
    columns = ["name", "recall_target", "pooled_threshold", "recall", "precision", "fp", "fall_clips_detected",
               "fall_clips", "non_fall_clips_alarmed", "non_fall_clips"]
    print(f"threshold trade off on pooled outer fold scores, read back from {outputs['tradeoff']}:")
    print(tradeoff[columns].round(3).to_string(index=False))
    for key in ("confusion", "pr", "importance_figure"):
        print(f"figure written: {outputs[key]}")


def main():
    parser = argparse.ArgumentParser(description="Figures and tables from training cross validation predictions.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    run = build_report(parser.parse_args().config)
    print_inputs(run)
    print_outputs(run)


if __name__ == "__main__":
    main()
