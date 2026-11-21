import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold, cross_val_predict

from src.features import FEATURE_NAMES
from src.models import (
    ML_MODELS,
    RULE_VARIANTS,
    build_pipeline,
    build_rule,
    fall_scores,
    param_grid,
    positive_column,
    score_method,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
METRICS = ["recall", "precision", "specificity", "f1", "accuracy"]
ML_VARIANTS = ("default", "tuned")
WINDOWLESS_COLUMNS = ["fall_clips_without_windows", "non_fall_clips_without_windows"]
RULE_THRESHOLD = 0.5
SCORE_FUNCTION = "src.models.fall_scores"
PREDICTION_KEYS = ["clip_id", "dataset", "t0", "t1", "label"]


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(cfg, key):
    return ROOT / cfg["paths"][key]


def load_training_windows(cfg):
    windows = pd.read_parquet(project_path(cfg, "windows"))
    windows = windows[windows["dataset"].isin(cfg["datasets"]["train"])].reset_index(drop=True)
    leaked = set(windows["dataset"]) & set(cfg["datasets"]["test"])
    if leaked:
        raise ValueError(f"test datasets in training windows: {sorted(leaked)}")
    return windows


def feature_columns(windows):
    absent = [name for name in FEATURE_NAMES if name not in windows.columns]
    if absent:
        raise ValueError(f"{len(absent)} features from src.features.FEATURE_NAMES missing in windows.parquet: {absent[:3]}")
    return list(FEATURE_NAMES)


def clip_fall_labels(cfg, windows):
    labels = pd.read_csv(project_path(cfg, "labels"))
    labels = labels[labels["dataset"].isin(cfg["datasets"]["train"])].set_index("clip_id")
    truth = labels["fall_start_s"].notna().astype(int)
    missing = set(windows["clip_id"]) - set(truth.index)
    if missing:
        raise ValueError(f"{len(missing)} window clips missing from labels.csv, e.g. {sorted(missing)[:3]}")
    positive_clips = set(windows.loc[windows["label"] == 1, "clip_id"])
    contradicted = sorted(clip for clip in positive_clips if truth[clip] == 0)
    if contradicted:
        raise ValueError(f"positive windows in non fall clips: {contradicted[:3]}")
    return truth


def clips_without_windows(cfg, windows):
    labels = pd.read_csv(project_path(cfg, "labels"))
    labels = labels[labels["dataset"].isin(cfg["datasets"]["train"])]
    fall = labels["fall_start_s"].notna()
    windowed = labels["clip_id"].isin(windows["clip_id"])
    counts = labels[["dataset"]].assign(
        labelled_clips=1,
        clips_with_windows=windowed,
        fall_clips_without_windows=fall & ~windowed,
        non_fall_clips_without_windows=~fall & ~windowed,
    )
    return counts.groupby("dataset").sum().astype(int).reset_index()


def assert_disjoint_groups(train, test):
    shared = set(train["clip_id"]) & set(test["clip_id"])
    if shared:
        raise AssertionError(f"clips on both sides of a split: {sorted(shared)[:3]}")


def outer_splits(windows, cfg):
    evaluation = cfg["evaluation"]
    positive_clips = windows.loc[windows["label"] == 1, "clip_id"].nunique()
    if positive_clips < evaluation["outer_folds"]:
        raise ValueError(
            f"only {positive_clips} clips have positive windows, need at least {evaluation['outer_folds']} for grouped CV"
        )
    splitter = StratifiedGroupKFold(
        n_splits=evaluation["outer_folds"], shuffle=True, random_state=evaluation["seed"]
    )
    return list(splitter.split(windows, windows["label"], windows["clip_id"]))


def inner_cv(cfg):
    return GroupKFold(n_splits=cfg["evaluation"]["inner_folds"])


def recall_threshold(y_true, scores, target):
    positive = scores[y_true == 1]
    candidates = np.unique(positive)[::-1]
    recalls = np.array([(positive >= candidate).mean() for candidate in candidates])
    return float(candidates[np.argmax(recalls >= target)])


def tuned_params(name, train, features, cfg):
    search = GridSearchCV(
        build_pipeline(name, cfg["models"], cfg["evaluation"]["seed"]),
        param_grid(name, cfg["models"]),
        scoring=cfg["evaluation"]["tuning_metric"],
        cv=inner_cv(cfg),
        n_jobs=cfg["models"]["n_jobs"],
    )
    search.fit(train[features], train["label"], groups=train["clip_id"])
    return search.best_params_


def out_of_fold_threshold(pipeline, train, features, cfg):
    scores = cross_val_predict(
        pipeline,
        train[features],
        train["label"],
        groups=train["clip_id"],
        cv=inner_cv(cfg),
        method=score_method(pipeline),
        n_jobs=cfg["models"]["n_jobs"],
    )
    return recall_threshold(train["label"].to_numpy(), positive_column(scores), cfg["evaluation"]["recall_target"])


def fit_ml(name, variant, train, features, cfg):
    pipeline = build_pipeline(name, cfg["models"], cfg["evaluation"]["seed"])
    params = tuned_params(name, train, features, cfg) if variant == "tuned" else {}
    pipeline.set_params(**params)
    threshold = out_of_fold_threshold(pipeline, train, features, cfg)
    pipeline.fit(train[features], train["label"])
    return pipeline, threshold, params


def fitted_attributes(model):
    hidden = {"n_features_in_", "feature_names_in_", "classes_"}
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in vars(model).items()
        if key.endswith("_") and key not in hidden
    }


def fold_predictions(train, test, features, cfg):
    for variant in RULE_VARIANTS:
        rule = build_rule(variant).fit(train[features], train["label"])
        scores = fall_scores(rule, test[features])
        yield "rule", variant, scores, rule.predict(test[features]), np.nan, fitted_attributes(rule)
    for name in ML_MODELS:
        for variant in ML_VARIANTS:
            model, threshold, params = fit_ml(name, variant, train, features, cfg)
            scores = fall_scores(model, test[features])
            yield name, variant, scores, (scores >= threshold).astype(int), threshold, params


def prediction_frame(test, base, scores, predicted):
    frame = test[PREDICTION_KEYS].reset_index(drop=True)
    return frame.assign(
        fold=base.get("fold", pd.NA),
        detector=base["detector"],
        variant=base["variant"],
        threshold=base["threshold"],
        score=np.asarray(scores, dtype=float),
        predicted=np.asarray(predicted, dtype=int),
    )


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "recall": ratio(tp, tp + fn),
        "precision": ratio(tp, tp + fp),
        "specificity": ratio(tn, tn + fp),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "accuracy": ratio(tp + tn, tp + tn + fp + fn),
        "n_test": int(tp + tn + fp + fn),
        "n_pos": int(tp + fn),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def clip_level(clip_ids, y_pred, truth):
    fired = pd.Series(np.asarray(y_pred), index=np.asarray(clip_ids)).groupby(level=0).max()
    return truth.loc[fired.index].to_numpy(), fired.to_numpy()


def score_rows(test, predicted, truth, base):
    clip_true, clip_pred = clip_level(test["clip_id"], predicted, truth)
    return [
        {**base, "level": "window", **binary_metrics(test["label"].to_numpy(), predicted)},
        {**base, "level": "clip", **binary_metrics(clip_true, clip_pred)},
    ]


def fold_counts(fold, train, test, truth):
    counts = {"fold": fold}
    for side, part in (("train", train), ("test", test)):
        clips = part["clip_id"].unique()
        counts[f"{side}_windows"] = len(part)
        counts[f"{side}_pos_windows"] = int(part["label"].sum())
        counts[f"{side}_clips"] = len(clips)
        counts[f"{side}_fall_clips"] = int(truth.loc[clips].sum())
    return counts


def summarize(results):
    summary = results.groupby(["detector", "variant", "level"])[METRICS].agg(["mean", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def with_windowless_counts(summary, windowless):
    clip_rows = summary["level"] == "clip"
    for column in WINDOWLESS_COLUMNS:
        summary[column] = pd.Series(pd.NA, index=summary.index, dtype="Int64")
        summary.loc[clip_rows, column] = int(windowless[column].sum())
    return summary


def with_pooled_clip_counting_windowless(summary, results, windowless):
    clips = results[results["level"] == "clip"].groupby(["detector", "variant"])[["tp", "fp", "fn"]].sum()
    tp, fp = clips["tp"], clips["fp"]
    fn = clips["fn"] + int(windowless["fall_clips_without_windows"].sum())
    pooled = pd.DataFrame({
        "pooled_clip_recall_windowless_missed": tp / (tp + fn),
        "pooled_clip_f1_windowless_missed": 2 * tp / (2 * tp + fp + fn),
    }).reset_index().assign(level="clip")
    return summary.merge(pooled, on=["detector", "variant", "level"], how="left")


def write_csv_atomic(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def selection_table(cv_path, target):
    results = pd.read_csv(cv_path)
    window = results[(results["level"] == "window") & results["detector"].isin(ML_MODELS)]
    table = window.groupby(["detector", "variant"]).agg(
        folds=("fold", "nunique"),
        recall_mean=("recall", "mean"),
        precision_mean=("precision", "mean"),
        f1_mean=("f1", "mean"),
    )
    table = table.reset_index()
    table.insert(0, "name", table["detector"] + "_" + table["variant"])
    table["reaches_recall_target"] = table["recall_mean"] >= target
    eligible = table[table["reaches_recall_target"]]
    pool = eligible if len(eligible) else table
    table["chosen"] = table.index == pool["f1_mean"].idxmax()
    return table


def save_joblib_atomic(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    joblib.dump(payload, temp)
    os.replace(temp, path)


def detector_file(model, threshold, name, features, windows, params):
    return {
        "model": model,
        "threshold": float(threshold),
        "name": name,
        "features": list(features),
        "score": SCORE_FUNCTION,
        "score_method": score_method(model),
        "params": params,
        "train_datasets": sorted(windows["dataset"].unique()),
        "train_windows": len(windows),
    }


def excluded_parquet_rows(cfg):
    datasets = pd.read_parquet(project_path(cfg, "windows"), columns=["dataset"])["dataset"]
    return datasets[~datasets.isin(cfg["datasets"]["train"])].value_counts().to_dict()


def freeze_detectors(cfg, windows, features, cv_path):
    target = cfg["evaluation"]["recall_target"]
    selection = selection_table(cv_path, target)
    selection_path = project_path(cfg, "tables") / "model_selection.csv"
    write_csv_atomic(selection, selection_path)
    chosen = selection[selection["chosen"]].iloc[0]
    model, threshold, params = fit_ml(chosen["detector"], chosen["variant"], windows, features, cfg)
    best = detector_file(model, threshold, chosen["name"], features, windows, params)
    rule = build_rule("tuned").fit(windows[features], windows["label"])
    rule_file = detector_file(rule, RULE_THRESHOLD, "rule_tuned", features, windows, fitted_attributes(rule))
    best_path, rule_path = ROOT / cfg["demo"]["model_file"], ROOT / cfg["demo"]["rule_file"]
    save_joblib_atomic(best, best_path)
    save_joblib_atomic(rule_file, rule_path)
    return {
        "selection": selection_path,
        "chosen": chosen.to_dict(),
        "recall_target": target,
        "best": best,
        "best_path": best_path,
        "rule": rule_file,
        "rule_path": rule_path,
        "test_rows_used": int(windows["dataset"].isin(cfg["datasets"]["test"]).sum()),
        "excluded_rows": excluded_parquet_rows(cfg),
    }


def cross_validate(config_path=CONFIG_PATH):
    cfg = load_config(config_path)
    windows = load_training_windows(cfg)
    features = feature_columns(windows)
    truth = clip_fall_labels(cfg, windows)
    windowless = clips_without_windows(cfg, windows)
    rows, counts, predictions = [], [], []
    for fold, (train_index, test_index) in enumerate(outer_splits(windows, cfg), 1):
        train, test = windows.iloc[train_index], windows.iloc[test_index]
        assert_disjoint_groups(train, test)
        counts.append(fold_counts(fold, train, test, truth))
        for detector, variant, scores, predicted, threshold, params in fold_predictions(train, test, features, cfg):
            base = {"detector": detector, "variant": variant, "fold": fold, "threshold": threshold, "best_params": str(params)}
            rows += score_rows(test, predicted, truth, base)
            predictions.append(prediction_frame(test, base, scores, predicted))
    results = pd.DataFrame(rows)
    results.insert(2, "level", results.pop("level"))
    tables = project_path(cfg, "tables")
    write_csv_atomic(results, tables / "cv_results.csv")
    summary = with_windowless_counts(summarize(results), windowless)
    write_csv_atomic(with_pooled_clip_counting_windowless(summary, results, windowless), tables / "tuning_before_after.csv")
    write_csv_atomic(windowless, tables / "clips_without_windows.csv")
    write_csv_atomic(pd.concat(predictions, ignore_index=True), tables / "oof_predictions.csv")
    return {
        "windows": windows,
        "features": features,
        "clip_truth": truth,
        "windowless": tables / "clips_without_windows.csv",
        "folds": pd.DataFrame(counts),
        "cv_results": tables / "cv_results.csv",
        "tuning": tables / "tuning_before_after.csv",
        "oof": tables / "oof_predictions.csv",
        "frozen": freeze_detectors(cfg, windows, features, tables / "cv_results.csv"),
    }


def frozen_detectors(cfg):
    best = joblib.load(ROOT / cfg["demo"]["model_file"])
    rule = joblib.load(ROOT / cfg["demo"]["rule_file"])
    training = load_training_windows(cfg)
    published = build_rule("published").fit(training[rule["features"]], training["label"])
    published_file = detector_file(
        published, RULE_THRESHOLD, "rule_published", rule["features"], training, fitted_attributes(published)
    )
    return [best, rule, published_file]


def activity_fall_truth(cfg, datasets):
    labels = pd.read_csv(project_path(cfg, "labels"))
    labels = labels[labels["dataset"].isin(datasets)]
    return labels.set_index("clip_id")["activity"].str.startswith("fall").astype(int)


def clip_predictions(clip_ids, predicted, truth):
    fired = pd.Series(np.asarray(predicted), index=np.asarray(clip_ids)).groupby(level=0).max()
    unknown = set(fired.index) - set(truth.index)
    if unknown:
        raise ValueError(f"{len(unknown)} window clips missing from labels.csv, e.g. {sorted(unknown)[:3]}")
    return fired.reindex(truth.index, fill_value=0).astype(int)


def exact_mcnemar(truth, first, second):
    first_right, second_right = first == truth, second == truth
    b = int((first_right & ~second_right).sum())
    c = int((~first_right & second_right).sum())
    return {
        "n": len(truth),
        "both_correct": int((first_right & second_right).sum()),
        "b_first_only_correct": b,
        "c_second_only_correct": c,
        "both_wrong": int((~first_right & ~second_right).sum()),
        "p_value": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0,
    }


def test_outputs(tables, datasets):
    prefix = "_".join(datasets)
    return {
        "results": tables / f"{prefix}_results.csv",
        "window_predictions": tables / f"{prefix}_window_predictions.csv",
        "clip_predictions": tables / f"{prefix}_clip_predictions.csv",
        "mcnemar": tables / f"{prefix}_mcnemar.csv",
    }


def score_frozen(detector, windows, truth):
    name = detector["name"]
    family, variant = name.rsplit("_", 1)
    scores = fall_scores(detector["model"], windows[detector["features"]])
    predicted = (scores >= detector["threshold"]).astype(int)
    base = {
        "detector": family,
        "variant": variant,
        "fold": pd.NA,
        "threshold": detector["threshold"],
        "best_params": str(detector["params"]),
    }
    clips = clip_predictions(windows["clip_id"], predicted, truth)
    rows = [
        {**base, "level": "window", **binary_metrics(windows["label"].to_numpy(), predicted)},
        {**base, "level": "clip", **binary_metrics(truth.to_numpy(), clips.to_numpy())},
    ]
    clip_frame = pd.DataFrame({"clip_id": truth.index, "detector": family, "variant": variant,
                               "label": truth.to_numpy(), "predicted": clips.to_numpy()})
    return rows, prediction_frame(windows, base, scores, predicted), clip_frame


def mcnemar_table(clip_frame, first, second):
    wide = clip_frame.assign(name=clip_frame["detector"] + "_" + clip_frame["variant"])
    wide = wide.pivot(index="clip_id", columns="name", values="predicted")
    truth = clip_frame.drop_duplicates("clip_id").set_index("clip_id")["label"].loc[wide.index]
    counts = exact_mcnemar(truth.to_numpy(), wide[first].to_numpy(), wide[second].to_numpy())
    return pd.DataFrame([{"level": "clip", "first": first, "second": second, **counts}])


def test_frozen(config_path=CONFIG_PATH, datasets=None, tables=None, force=False):
    cfg = load_config(config_path)
    datasets = list(datasets or cfg["datasets"]["test"])
    outputs = test_outputs(Path(tables) if tables else project_path(cfg, "tables"), datasets)
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(f"already written, the test runs once: {existing}")
    detectors = frozen_detectors(cfg)
    overlap = sorted(set(datasets) & {name for d in detectors for name in d["train_datasets"]})
    if overlap:
        raise ValueError(f"frozen detectors were trained on {overlap}, refusing to test on them")
    windows = pd.read_parquet(project_path(cfg, "windows"))
    windows = windows[windows["dataset"].isin(datasets)].reset_index(drop=True)
    truth = activity_fall_truth(cfg, datasets)
    scored = [score_frozen(detector, windows, truth) for detector in detectors]
    results = pd.DataFrame([row for rows, _, _ in scored for row in rows])
    results.insert(2, "level", results.pop("level"))
    clip_frame = pd.concat([clips for _, _, clips in scored], ignore_index=True)
    mcnemar = mcnemar_table(clip_frame, detectors[0]["name"], detectors[1]["name"])
    write_csv_atomic(results, outputs["results"])
    write_csv_atomic(pd.concat([frame for _, frame, _ in scored], ignore_index=True), outputs["window_predictions"])
    write_csv_atomic(clip_frame, outputs["clip_predictions"])
    write_csv_atomic(mcnemar, outputs["mcnemar"])
    return {
        "datasets": datasets,
        "outputs": outputs,
        "windows": windows,
        "truth": truth,
        "windowless": truth[~truth.index.isin(windows["clip_id"])],
        "detectors": [{key: detector[key] for key in ("name", "threshold", "params", "train_datasets")} for detector in detectors],
        "trained_on_tested": sorted(set(datasets) & {name for d in detectors for name in d["train_datasets"]}),
    }
