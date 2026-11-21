import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import features

ID_COLUMNS = ["clip_id", "dataset", "label", "t0", "t1"]
EPS = 1e-9
MOSTLY_NAN_FRACTION = 0.5


def joint_on_grid(t, xy, grid, max_gap_s):
    after = np.searchsorted(t, grid, side="left")
    before = np.searchsorted(t, grid, side="right") - 1
    inside = (before >= 0) & (after < t.size)
    gap = t[np.minimum(after, t.size - 1)] - t[np.maximum(before, 0)]
    values = np.column_stack([np.interp(grid, t, xy[:, 0]), np.interp(grid, t, xy[:, 1])])
    values[~(inside & (gap <= max_gap_s + EPS))] = np.nan
    return values


def resample(t, kp, grid, max_gap_s):
    xy = np.asarray(kp, dtype=float)[..., :2]
    out = np.full((grid.size, xy.shape[1], 2), np.nan)
    for joint in range(xy.shape[1]):
        valid = ~np.isnan(xy[:, joint]).any(axis=1)
        if valid.any():
            out[:, joint] = joint_on_grid(t[valid], xy[valid, joint], grid, max_gap_s)
    return out


def window_starts(t, fps, length_s, stride_s):
    span = t[-1] + 1.0 / fps - t[0]
    count = max(0, int(np.floor((span - length_s) / stride_s + EPS)) + 1)
    return t[0] + stride_s * np.arange(count)


def falling_coverage(t0, t1, fall_start, fall_end):
    if np.isnan(fall_start) or np.isnan(fall_end):
        return 0.0
    return max(0.0, min(t1, fall_end) - max(t0, fall_start)) / (fall_end - fall_start)


def window_label(coverage, positive_min_overlap):
    if coverage >= positive_min_overlap - EPS:
        return 1
    return None if coverage > EPS else 0


def window_grid(t0, settings):
    frames = int(round(settings["length_s"] * settings["rate_hz"]))
    return t0 + np.arange(frames) / settings["rate_hz"]


def clip_windows(t, kp, fps, fall_start, fall_end, settings):
    kept, ambiguous, missing_pose = [], 0, 0
    starts = window_starts(t, fps, settings["length_s"], settings["stride_s"])
    for t0 in starts:
        t1 = t0 + settings["length_s"]
        label = window_label(falling_coverage(t0, t1, fall_start, fall_end), settings["positive_min_overlap"])
        if label is None:
            ambiguous += 1
            continue
        xy = resample(t, kp, window_grid(t0, settings), settings["max_gap_fill_s"])
        if np.mean(~features.usable_frames(xy)) > settings["max_missing_fraction"] + EPS:
            missing_pose += 1
            continue
        kept.append((t0, t1, label, xy))
    return kept, ambiguous, missing_pose, len(starts)


def save_outputs(table, sequences, paths):
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    table_tmp = paths["windows"].with_name(paths["windows"].name + ".tmp")
    table.to_parquet(table_tmp, index=False)
    os.replace(table_tmp, paths["windows"])
    sequences_tmp = paths["sequences"].with_name(paths["sequences"].name + ".tmp")
    with open(sequences_tmp, "wb") as handle:
        np.savez(handle, **sequences)
    os.replace(sequences_tmp, paths["sequences"])


def build(config_path):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.parent
    settings = config["windows"]
    last_s = config["baseline"]["last_s"]
    labels = pd.read_csv(root / config["paths"]["labels"])
    keypoint_dir = root / config["paths"]["keypoints"]
    rows, sequences = [], []
    clips_read, clips_without_keypoints = Counter(), Counter()
    ambiguous, missing_pose, zero_window_clips = 0, 0, []
    for clip in labels.itertuples(index=False):
        path = keypoint_dir / f"{clip.clip_id}.npz"
        if not path.exists():
            clips_without_keypoints[clip.dataset] += 1
            continue
        with np.load(path) as data:
            t, kp, fps = data["t"].astype(float), data["kp"], float(data["fps"])
        clips_read[clip.dataset] += 1
        kept, clip_ambiguous, clip_missing, candidates = clip_windows(
            t, kp, fps, float(clip.fall_start_s), float(clip.fall_end_s), settings
        )
        ambiguous += clip_ambiguous
        missing_pose += clip_missing
        if not kept:
            reason = "shorter than one window" if candidates == 0 else "every window dropped"
            zero_window_clips.append((clip.clip_id, len(t), reason))
        for t0, t1, label, xy in kept:
            row = {"clip_id": clip.clip_id, "dataset": clip.dataset, "label": label, "t0": t0, "t1": t1}
            row.update(features.window_features(xy, settings["rate_hz"], last_s))
            rows.append(row)
            sequences.append(features.sequence(xy))
    table = pd.DataFrame(rows, columns=ID_COLUMNS + features.FEATURE_NAMES)
    table["label"] = table["label"].astype(int)
    frames = int(round(settings["length_s"] * settings["rate_hz"]))
    stacked = np.array(sequences, dtype=float).reshape(-1, frames, 2 * len(features.SEQUENCE_JOINTS))
    save_outputs(
        table,
        {
            "X": np.nan_to_num(stacked, nan=0.0).astype(np.float32),
            "missing": np.isnan(stacked),
            "y": table["label"].to_numpy(dtype=np.int64),
            "clip_id": table["clip_id"].to_numpy(dtype=str),
            "dataset": table["dataset"].to_numpy(dtype=str),
            "t0": table["t0"].to_numpy(dtype=float),
            "t1": table["t1"].to_numpy(dtype=float),
        },
        {"windows": root / config["paths"]["windows"], "sequences": root / config["paths"]["sequences"]},
    )
    train = table[table["dataset"].isin(config["datasets"]["train"])]
    nan_fraction = train[features.FEATURE_NAMES].isna().mean() if len(train) else pd.Series(dtype=float)
    return {
        "clips_read": dict(clips_read),
        "clips_without_keypoints": dict(clips_without_keypoints),
        "windows_per_class": table["label"].value_counts().sort_index().to_dict(),
        "windows_per_dataset_and_class": pd.crosstab(table["dataset"], table["label"]),
        "dropped_ambiguous": ambiguous,
        "dropped_missing_pose": missing_pose,
        "zero_window_clips": zero_window_clips,
        "feature_count": len(features.FEATURE_NAMES),
        "sequence_shape": stacked.shape,
        "nan_fraction": nan_fraction[nan_fraction > 0].sort_values(ascending=False),
        "mostly_nan": list(nan_fraction[nan_fraction > MOSTLY_NAN_FRACTION].index),
    }
