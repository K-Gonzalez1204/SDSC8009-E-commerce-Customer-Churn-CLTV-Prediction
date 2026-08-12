import warnings
from contextlib import contextmanager

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .config import CV_FOLDS, K_BEST, SEED
from .preprocess import build_preprocessor


@contextmanager
def _suppress_lgbm_name_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier",
        )
        yield


def build_model_specs(pos_weight: float):
    return {
        "LogisticRegression": (
            lambda: LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=SEED
            )
        ),
        "RandomForest": (
            lambda: RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            )
        ),
        "XGBoost": (
            lambda: XGBClassifier(
                n_estimators=100,
                scale_pos_weight=pos_weight,
                random_state=SEED,
                eval_metric="logloss",
                verbosity=0,
            )
        ),
        "LightGBM": (
            lambda: LGBMClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=SEED,
                verbose=-1,
            )
        ),
    }


def build_pipelines(pos_weight: float, k_best: int = K_BEST):
    specs = build_model_specs(pos_weight)
    pipes = {
        name: Pipeline(
            [("pre", build_preprocessor(k_best=k_best)), ("model", factory())]
        )
        for name, factory in specs.items()
    }
    pipes["Voting"] = Pipeline(
        [
            ("pre", build_preprocessor(k_best=k_best)),
            (
                "model",
                VotingClassifier(
                    estimators=[
                        (name, clone(pipes[name].steps[-1][1])) for name in specs
                    ],
                    voting="soft",
                ),
            ),
        ]
    )
    return pipes


def run_cv_with_oof(pipelines, X, y):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scores = {}
    oof_probs = {}
    for name, pipe in pipelines.items():
        proba = np.zeros(len(y))
        fold_scores = []
        for train_idx, valid_idx in cv.split(X, y):
            fitted = clone(pipe)
            with _suppress_lgbm_name_warning():
                fitted.fit(X[train_idx], y[train_idx])
                p = fitted.predict_proba(X[valid_idx])[:, 1]
            proba[valid_idx] = p
            fold_scores.append(roc_auc_score(y[valid_idx], p))
        scores[name] = np.asarray(fold_scores)
        oof_probs[name] = proba
    return scores, oof_probs


def select_best(scores, preferred: str = "LogisticRegression") -> str:
    means = {name: float(np.mean(values)) for name, values in scores.items()}
    best = max(means, key=means.get)
    if abs(means[best] - means.get(preferred, -1.0)) < 1e-3:
        return preferred
    return best


def select_threshold(y_true, y_prob) -> float:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.round(np.arange(0.10, 0.91, 0.05), 2):
        pred = (y_prob >= thr).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)
    return best_thr


def evaluate_holdout(pipeline, X_train, y_train, X_test, y_test, threshold):
    fitted = clone(pipeline)
    with _suppress_lgbm_name_warning():
        fitted.fit(X_train, y_train)
        prob = fitted.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, labels=[0, 1], average=None, zero_division=0
    )
    metrics = {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "pr_auc": float(average_precision_score(y_test, prob)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "confusion_matrix": confusion_matrix(
            y_test, pred, labels=[0, 1]
        ).tolist(),
        "test_size": int(len(y_test)),
    }
    return metrics, prob


def bootstrap_auc(y_true, y_prob, seed: int = SEED, n: int = 1000):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    values = []
    index = np.arange(len(y_true))
    for _ in range(n):
        sample = rng.choice(index, size=len(index), replace=True)
        if len(np.unique(y_true[sample])) < 2:
            continue
        values.append(roc_auc_score(y_true[sample], y_prob[sample]))
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
