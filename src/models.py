from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.baseline import RuleDetector

ML_MODELS = ("nb", "rf", "svm")
RULE_VARIANTS = ("tuned", "published")


def build_pipeline(name, models_cfg, seed):
    imputer = ("impute", SimpleImputer(strategy=models_cfg["imputer_strategy"]))
    if name == "nb":
        return Pipeline([imputer, ("clf", GaussianNB())])
    if name == "rf":
        forest = RandomForestClassifier(
            n_estimators=models_cfg["rf"]["n_estimators"],
            class_weight="balanced",
            random_state=seed,
            n_jobs=models_cfg["n_jobs"],
        )
        return Pipeline([imputer, ("clf", forest)])
    if name == "svm":
        svc = SVC(kernel="rbf", class_weight="balanced", random_state=seed)
        return Pipeline([imputer, ("scale", StandardScaler()), ("clf", svc)])
    raise ValueError(f"unknown model {name}")


def param_grid(name, models_cfg):
    return {f"clf__{key}": values for key, values in models_cfg[name]["grid"].items()}


def build_rule(variant):
    return RuleDetector(tune=variant == "tuned")


def score_method(model):
    return "predict_proba" if hasattr(model, "predict_proba") else "decision_function"


def positive_column(scores):
    return scores[:, 1] if scores.ndim == 2 else scores


def fall_scores(model, X):
    return positive_column(getattr(model, score_method(model))(X))
