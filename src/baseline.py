from itertools import product
from pathlib import Path

import numpy as np
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.utils.validation import check_is_fitted

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
RATIO_COLUMN = "bbox_ratio_last"
ANGLE_COLUMN = "trunk_angle_last"
SPEED_COLUMN = "hip_down_speed_max"


def rule_decision(X, ratio, angle_deg, speed):
    wide = X[RATIO_COLUMN].to_numpy(dtype=float) > ratio
    tilted = X[ANGLE_COLUMN].to_numpy(dtype=float) > angle_deg
    fast = X[SPEED_COLUMN].to_numpy(dtype=float) > speed
    return (wide & tilted & fast).astype(int)


def best_thresholds(X, y, settings):
    grid = product(settings["grid_ratio"], settings["grid_angle_deg"], settings["grid_speed_tl_per_s"])
    scored = [(f1_score(y, rule_decision(X, *combo), zero_division=0), combo) for combo in grid]
    best_f1, best_combo = max(scored, key=lambda item: item[0])
    return best_combo, best_f1


class RuleDetector(ClassifierMixin, BaseEstimator):
    def __init__(self, tune=True, ratio=None, angle_deg=None, speed=None):
        self.tune = tune
        self.ratio = ratio
        self.angle_deg = angle_deg
        self.speed = speed

    def fit(self, X, y):
        settings = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["baseline"]
        y = np.asarray(y).astype(int)
        if self.tune:
            (self.ratio_, self.angle_deg_, self.speed_), self.train_f1_ = best_thresholds(X, y, settings)
        else:
            self.ratio_ = settings["published_ratio"] if self.ratio is None else self.ratio
            self.angle_deg_ = settings["published_angle_deg"] if self.angle_deg is None else self.angle_deg
            self.speed_ = settings["default_speed_tl_per_s"] if self.speed is None else self.speed
            self.train_f1_ = f1_score(y, rule_decision(X, self.ratio_, self.angle_deg_, self.speed_), zero_division=0)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        check_is_fitted(self, "classes_")
        return rule_decision(X, self.ratio_, self.angle_deg_, self.speed_)

    def predict_proba(self, X):
        decision = self.predict(X).astype(float)
        return np.column_stack([1.0 - decision, decision])
