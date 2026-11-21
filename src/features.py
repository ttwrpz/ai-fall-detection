import numpy as np

HEAD = [0, 7, 8]
SHOULDERS = [11, 12]
HIPS = [23, 24]
ANKLES = [27, 28]
SEQUENCE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
SERIES = (
    "trunk_angle",
    "trunk_ang_vel",
    "hip_down_speed",
    "hip_down_accel",
    "head_hip_gap",
    "hip_ankle_gap",
    "bbox_ratio",
)
STATS = ("min", "max", "mean", "std", "range")
LAST_MEAN_SERIES = ("trunk_angle", "head_hip_gap", "hip_ankle_gap", "bbox_ratio")
FEATURE_NAMES = [
    f"{name}_{stat}"
    for name in SERIES
    for stat in STATS + (("last",) if name in LAST_MEAN_SERIES else ())
]


def midpoint(xy, joints):
    part = xy[:, joints]
    count = np.sum(~np.isnan(part), axis=1)
    with np.errstate(invalid="ignore"):
        return np.nansum(part, axis=1) / count


def valid_median(values):
    values = values[~np.isnan(values)]
    return np.median(values) if values.size else np.nan


def usable_frames(xy):
    return ~np.isnan(midpoint(xy, HIPS)[:, 1]) & ~np.isnan(midpoint(xy, SHOULDERS)[:, 1])


def normalize(xy):
    hip = midpoint(xy, HIPS)
    torso = np.linalg.norm(midpoint(xy, SHOULDERS) - hip, axis=1)
    scale = valid_median(torso)
    if not scale > 0:
        return np.full_like(xy, np.nan, dtype=float)
    center = np.array([valid_median(hip[:, 0]), valid_median(hip[:, 1])])
    return (xy - center) / scale


def bbox_ratio(xy):
    width = np.fmax.reduce(xy[..., 0], axis=1) - np.fmin.reduce(xy[..., 0], axis=1)
    height = np.fmax.reduce(xy[..., 1], axis=1) - np.fmin.reduce(xy[..., 1], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(height > 0, width / height, np.nan)


def frame_series(xy, rate_hz):
    dt = 1.0 / rate_hz
    hip = midpoint(xy, HIPS)
    trunk = midpoint(xy, SHOULDERS) - hip
    trunk_angle = np.degrees(np.arctan2(np.abs(trunk[:, 0]), -trunk[:, 1]))
    hip_down_speed = np.gradient(hip[:, 1], dt)
    return {
        "trunk_angle": trunk_angle,
        "trunk_ang_vel": np.gradient(trunk_angle, dt),
        "hip_down_speed": hip_down_speed,
        "hip_down_accel": np.gradient(hip_down_speed, dt),
        "head_hip_gap": hip[:, 1] - midpoint(xy, HEAD)[:, 1],
        "hip_ankle_gap": midpoint(xy, ANKLES)[:, 1] - hip[:, 1],
        "bbox_ratio": bbox_ratio(xy),
    }


def series_stats(values):
    valid = values[~np.isnan(values)]
    if not valid.size:
        return dict.fromkeys(STATS, np.nan)
    low, high = valid.min(), valid.max()
    return {"min": low, "max": high, "mean": valid.mean(), "std": valid.std(), "range": high - low}


def summarize(series, rate_hz, last_s):
    last_count = max(1, int(round(last_s * rate_hz)))
    summary = {}
    for name, values in series.items():
        summary.update({f"{name}_{stat}": value for stat, value in series_stats(values).items()})
        if name in LAST_MEAN_SERIES:
            tail = values[-last_count:]
            tail = tail[~np.isnan(tail)]
            summary[f"{name}_last"] = tail.mean() if tail.size else np.nan
    return summary


def window_features(xy, rate_hz, last_s):
    return summarize(frame_series(normalize(xy), rate_hz), rate_hz, last_s)


def sequence(xy):
    return normalize(xy)[:, SEQUENCE_JOINTS].reshape(len(xy), -1)
