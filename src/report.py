import csv
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from src.evaluate import (  # noqa: E402
    CONFIG_PATH, ROOT, binary_metrics, load_config, project_path, summarize, write_csv_atomic,
)

RULE_NAMES = ["rule_published", "rule_tuned"]
NO_ACTIVITY_NAMES = "none in dataset files, grouped by dataset"
GMDCSA24_SOURCE = "GMDCSA-24 ADL.csv Description"


def load_oof(cfg):
    oof = pd.read_csv(project_path(cfg, "tables") / "oof_predictions.csv")
    leaked = set(oof["dataset"]) - set(cfg["datasets"]["train"])
    if leaked:
        raise ValueError(f"non training datasets in out of fold predictions: {sorted(leaked)}")
    return oof.assign(name=oof["detector"] + "_" + oof["variant"])


def selected_per_model(cfg):
    selection = pd.read_csv(project_path(cfg, "tables") / "model_selection.csv")
    picks = []
    for _, group in selection.groupby("detector", sort=False):
        eligible = group[group["reaches_recall_target"]]
        pool = eligible if len(eligible) else group
        picks.append(pool.loc[pool["f1_mean"].idxmax(), "name"])
    return picks


def pooled_metrics(oof, names):
    rows = []
    for name in names:
        part = oof[oof["name"] == name]
        average_precision = float("nan") if name in RULE_NAMES else average_precision_score(part["label"], part["score"])
        rows.append({"name": name, "folds": part["fold"].nunique(), **binary_metrics(part["label"], part["predicted"]),
                     "average_precision": average_precision})
    return pd.DataFrame(rows)


def plot_confusion(pooled, cfg, path, title, size_key="confusion_size_in"):
    settings = cfg["report"]
    figure, axes = plt.subplots(1, len(pooled), figsize=settings[size_key])
    for axis, row in zip(axes, pooled.itertuples()):
        counts = [[row.tn, row.fp], [row.fn, row.tp]]
        sns.heatmap(counts, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis,
                    xticklabels=["not fall", "fall"], yticklabels=["not fall", "fall"])
        axis.set_title(f"{row.name}\nrecall {row.recall:.3f}, F1 {row.f1:.3f}")
        axis.set_xlabel("predicted")
        axis.set_ylabel("true")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=settings["dpi"])
    plt.close(figure)


def dataset_names(cfg, role):
    return " + ".join(cfg["report"]["dataset_titles"][name] for name in cfg["datasets"][role])


def dataset_summary(cfg):
    labels = pd.read_csv(project_path(cfg, "labels"))
    windows = pd.read_parquet(project_path(cfg, "windows"), columns=["clip_id", "dataset", "label"])
    roles = {name: role for role in ("train", "test") for name in cfg["datasets"][role]}
    clips = labels.assign(fall=labels["activity"].str.startswith("fall"), windowed=labels["clip_id"].isin(windows["clip_id"]))
    table = clips.groupby("dataset", sort=False).agg(
        clips=("clip_id", "size"), fall_clips=("fall", "sum"), subjects=("subject", "nunique"),
        fps_min=("fps", "min"), fps_max=("fps", "max"), clips_with_windows=("windowed", "sum"),
    )
    table["non_fall_clips"] = table["clips"] - table["fall_clips"]
    table["windows"] = windows.groupby("dataset").size()
    table["fall_windows"] = windows.groupby("dataset")["label"].sum()
    table["role"] = [roles.get(name, "unused") for name in table.index]
    table["subjects"] = table["subjects"].astype("Int64").mask(table["subjects"] == 0)
    columns = ["role", "clips", "fall_clips", "non_fall_clips", "subjects", "fps_min", "fps_max",
               "clips_with_windows", "windows", "fall_windows"]
    return table[columns].reset_index()


def lstm_summary(cfg):
    lstm = pd.read_csv(project_path(cfg, "tables") / "lstm_cv_results.csv")
    summary = summarize(lstm)
    summary.insert(3, "folds", lstm.groupby(["detector", "variant", "level"])["fold"].nunique().to_numpy())
    return summary


def fold_operating_points(part):
    return pd.DataFrame([
        {"fold": fold, "threshold": group["threshold"].iloc[0],
         **binary_metrics(group["label"], group["predicted"])}
        for fold, group in part.groupby("fold")
    ])


def plot_pr_curves(oof, pooled, names, cfg, path):
    settings = cfg["report"]
    target = cfg["evaluation"]["recall_target"]
    figure, axes = plt.subplots(1, len(names), figsize=settings["pr_size_in"], sharey=True)
    for axis, name in zip(axes, names):
        part = oof[oof["name"] == name]
        summary = pooled.set_index("name").loc[name]
        precision, recall, _ = precision_recall_curve(part["label"], part["score"])
        points = fold_operating_points(part)
        axis.plot(recall, precision, label=f"pooled curve, AP {summary['average_precision']:.3f}")
        axis.scatter(points["recall"], points["precision"], marker="o", color="tab:orange", zorder=3,
                     label=f"fold thresholds ({points['threshold'].min():.3g} to {points['threshold'].max():.3g})")
        axis.scatter([summary["recall"]], [summary["precision"]], marker="*", s=200, color="tab:red", zorder=4,
                     label=f"pooled operating point (recall {summary['recall']:.3f})")
        axis.axvline(target, linestyle="--", color="grey", label=f"recall target {target}")
        axis.set_title(name)
        axis.set_xlabel("recall")
        axis.legend(loc="lower left", fontsize="small")
    axes[0].set_ylabel("precision")
    figure.suptitle(f"Precision recall, outer fold window scores, pooled across fold models ({dataset_names(cfg, 'train')})")
    figure.tight_layout()
    figure.savefig(path, dpi=settings["dpi"])
    plt.close(figure)


def forest_importance(cfg):
    best = joblib.load(ROOT / cfg["demo"]["model_file"])
    forest = best["model"].named_steps["clf"]
    if not hasattr(forest, "feature_importances_"):
        raise ValueError(f"{best['name']} in {cfg['demo']['model_file']} has no impurity feature importance")
    table = pd.DataFrame({"feature": best["features"], "importance": forest.feature_importances_})
    table = table.sort_values("importance", ascending=False, ignore_index=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table.assign(model=best["name"], method="impurity (mean decrease in Gini), refit on all training windows")


def plot_importance(table, cfg, path):
    settings = cfg["report"]
    top = table.head(settings["top_n_features"]).iloc[::-1]
    figure, axis = plt.subplots(figsize=settings["importance_size_in"])
    axis.barh(top["feature"], top["importance"], color="tab:green")
    axis.set_xlabel("impurity importance")
    axis.set_title(f"{table['model'].iloc[0]}: top {len(top)} features")
    figure.tight_layout()
    figure.savefig(path, dpi=settings["dpi"])
    plt.close(figure)


def gmdcsa24_adl_activities(cfg):
    activities = {}
    for path in sorted((ROOT / cfg["paths"]["raw"] / "gmdcsa24" / "files").glob("*/Subject */ADL.csv")):
        subject = int(path.parent.name.split()[-1])
        with open(path, newline="", encoding="utf-8-sig") as handle:
            rows = [row for row in csv.reader(handle) if row]
        column = [name.strip() for name in rows[0]].index("Description")
        for row in rows[1:]:
            clip_id = f"gmdcsa24-s{subject}-adl-{int(Path(row[0].strip()).stem):02d}"
            activities[clip_id] = ",".join(row[column:]).strip(" ,")
    return activities


def non_fall_activities(cfg):
    labels = pd.read_csv(project_path(cfg, "labels"))
    labels = labels[labels["dataset"].isin(cfg["datasets"]["train"]) & ~labels["activity"].str.startswith("fall")]
    described = gmdcsa24_adl_activities(cfg)
    found = labels["clip_id"].map(described)
    activity = found.fillna(labels["dataset"] + " " + labels["activity"])
    source = found.notna().map({True: GMDCSA24_SOURCE, False: NO_ACTIVITY_NAMES})
    return labels.assign(activity=activity, activity_source=source)[["clip_id", "dataset", "activity", "activity_source"]]


def alarm_counts(oof, clips, name):
    part = oof[oof["name"] == name].merge(clips, on="clip_id", suffixes=("", "_label"))
    per_clip = part.groupby(["dataset", "activity", "activity_source", "clip_id"])["predicted"].agg(["max", "sum", "size"])
    keys = ["dataset", "activity", "activity_source"]
    totals = per_clip.groupby(level=keys).agg(clips=("max", "size"), clips_alarmed=("max", "sum"),
                                              windows=("size", "sum"), windows_alarmed=("sum", "sum"))
    dataset_totals = per_clip.groupby(level="dataset").agg(clips=("max", "size"), clips_alarmed=("max", "sum"),
                                                           windows=("size", "sum"), windows_alarmed=("sum", "sum"))
    dataset_totals.index = pd.MultiIndex.from_tuples(
        [(dataset, "all non fall clips", "dataset total") for dataset in dataset_totals.index], names=keys
    )
    counts = pd.concat([totals, dataset_totals])
    counts["clip_alarm_rate"] = counts["clips_alarmed"] / counts["clips"]
    counts["window_alarm_rate"] = counts["windows_alarmed"] / counts["windows"]
    return counts


def activity_alarm_rates(oof, clips, names):
    per_name = [alarm_counts(oof, clips, name) for name in names]
    table = per_name[0][["clips", "windows"]].copy()
    for name, counts in zip(names, per_name):
        if not counts[["clips", "windows"]].equals(table[["clips", "windows"]]):
            raise ValueError(f"{name} covers different clips or windows than {names[0]}")
        for column in ["clips_alarmed", "clip_alarm_rate", "windows_alarmed", "window_alarm_rate"]:
            table[f"{name}_{column}"] = counts[column]
    table = table.reset_index()
    order = [f"{names[-1]}_clip_alarm_rate", f"{names[-1]}_window_alarm_rate", f"{names[0]}_clip_alarm_rate"]
    is_total = table["activity_source"] == "dataset total"
    return table.assign(is_total=is_total).sort_values(
        ["is_total", "dataset", *order], ascending=[False, True, False, False, False], ignore_index=True
    ).drop(columns="is_total")


def fall_times(cfg):
    labels = pd.read_csv(project_path(cfg, "labels"))
    labels = labels[labels["dataset"].isin(cfg["datasets"]["train"]) & labels["fall_start_s"].notna()]
    return labels.set_index("clip_id")[["fall_start_s", "fall_end_s"]]


def cv_detection_delay(oof, names, cfg):
    times = fall_times(cfg)
    rows = []
    for name in names:
        part = oof[(oof["name"] == name) & oof["clip_id"].isin(times.index)].join(times, on="clip_id")
        after_onset = part[(part["predicted"] == 1) & (part["t1"] > part["fall_start_s"])]
        first = after_onset.groupby("clip_id")["t1"].min()
        for clip_id, (start, end) in times.loc[times.index.isin(part["clip_id"])].iterrows():
            fired = clip_id in first.index
            rows.append({"name": name, "clip_id": clip_id, "fall_start_s": start, "fall_end_s": end,
                         "first_positive_window_end_s": first.get(clip_id), "detected": fired,
                         "delay_from_start_s": first[clip_id] - start if fired else None,
                         "delay_from_end_s": first[clip_id] - end if fired else None})
    return pd.DataFrame(rows)


def delay_summary(delays):
    detected = delays[delays["detected"]]
    summary = delays.groupby("name", sort=False).agg(fall_clips=("clip_id", "size"), detected=("detected", "sum"))
    for column in ("delay_from_start_s", "delay_from_end_s"):
        grouped = detected.groupby("name", sort=False)[column]
        summary[f"{column}_median"] = grouped.median()
        summary[f"{column}_p90"] = grouped.quantile(0.9)
        summary[f"{column}_max"] = grouped.max()
    return summary.reset_index()


def clip_rates(part, predicted):
    fired = part.assign(predicted=predicted).groupby("clip_id").agg(fired=("predicted", "max"), fall=("is_fall", "max"))
    falls, others = fired[fired["fall"]], fired[~fired["fall"]]
    return int(falls["fired"].sum()), len(falls), int(others["fired"].sum()), len(others)


def threshold_tradeoff(oof, names, cfg):
    fall_clips = set(fall_times(cfg).index)
    rows = []
    for name in names:
        part = oof[oof["name"] == name].assign(is_fall=lambda frame: frame["clip_id"].isin(fall_clips))
        precision, recall, thresholds = precision_recall_curve(part["label"], part["score"])
        for target in cfg["report"]["tradeoff_recall_targets"]:
            threshold = thresholds[recall[:-1] >= target].max()
            predicted = (part["score"] >= threshold).astype(int)
            detected, falls, alarmed, others = clip_rates(part, predicted)
            rows.append({"name": name, "recall_target": target, "pooled_threshold": threshold,
                         **binary_metrics(part["label"], predicted), "fall_clips_detected": detected,
                         "fall_clips": falls, "non_fall_clips_alarmed": alarmed, "non_fall_clips": others})
    return pd.DataFrame(rows)


def build_report(config_path=CONFIG_PATH):
    cfg = load_config(config_path)
    tables, figures = project_path(cfg, "tables"), project_path(cfg, "figures")
    figures.mkdir(parents=True, exist_ok=True)
    oof = load_oof(cfg)
    models = selected_per_model(cfg)
    pooled = pooled_metrics(oof, RULE_NAMES + models)
    importance = forest_importance(cfg)
    clips = non_fall_activities(cfg)
    alarm_names = [joblib.load(ROOT / cfg["demo"][key])["name"] for key in ("rule_file", "model_file")]
    alarms = activity_alarm_rates(oof, clips, alarm_names)
    outputs = {
        "pooled": tables / "cv_pooled_metrics.csv",
        "importance": tables / "rf_feature_importance.csv",
        "alarms": tables / "activity_alarm_rates.csv",
        "datasets": tables / "dataset_summary.csv",
        "lstm": tables / "lstm_summary.csv",
        "delay": tables / "cv_detection_delay.csv",
        "delay_summary": tables / "cv_detection_delay_summary.csv",
        "tradeoff": tables / "cv_threshold_tradeoff.csv",
        "confusion": figures / "confusion_matrices_cv.png",
        "pr": figures / "pr_curves_cv.png",
        "importance_figure": figures / "rf_feature_importance.png",
    }
    write_csv_atomic(pooled, outputs["pooled"])
    write_csv_atomic(importance, outputs["importance"])
    write_csv_atomic(alarms, outputs["alarms"])
    write_csv_atomic(dataset_summary(cfg), outputs["datasets"])
    write_csv_atomic(lstm_summary(cfg), outputs["lstm"])
    delays = cv_detection_delay(oof, RULE_NAMES + models, cfg)
    write_csv_atomic(delays, outputs["delay"])
    write_csv_atomic(delay_summary(delays), outputs["delay_summary"])
    write_csv_atomic(threshold_tradeoff(oof, models, cfg), outputs["tradeoff"])
    title = f"Window level, pooled outer folds, per fold frozen thresholds ({dataset_names(cfg, 'train')})"
    plot_confusion(pooled, cfg, outputs["confusion"], title)
    plot_pr_curves(oof, pooled, models, cfg, outputs["pr"])
    plot_importance(importance, cfg, outputs["importance_figure"])
    return {
        "oof": oof,
        "models": models,
        "alarm_names": alarm_names,
        "clips": clips,
        "outputs": outputs,
        "top_n": cfg["report"]["top_n_features"],
    }


def build_test_report(config_path=CONFIG_PATH):
    cfg = load_config(config_path)
    prefix = "_".join(cfg["datasets"]["test"])
    results = pd.read_csv(project_path(cfg, "tables") / f"{prefix}_results.csv")
    results = results.assign(name=results["detector"] + "_" + results["variant"])
    figures = project_path(cfg, "figures")
    figures.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for level in ("window", "clip"):
        rows = results[results["level"] == level].reset_index(drop=True)
        title = f"{level.capitalize()} level, frozen thresholds, single test run ({dataset_names(cfg, 'test')})"
        outputs[level] = figures / f"{prefix}_confusion_{level}.png"
        plot_confusion(rows, cfg, outputs[level], title, "test_confusion_size_in")
    return {"results": results, "outputs": outputs}
