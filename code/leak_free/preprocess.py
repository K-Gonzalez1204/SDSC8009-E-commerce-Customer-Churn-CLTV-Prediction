import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .config import K_BEST


class QuantileClipTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.quantile(X, self.lower, axis=0)
        self.upper_bounds_ = np.quantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def set_output(self, transform=None):
        return self


def build_preprocessor(k_best: int = K_BEST) -> Pipeline:
    return Pipeline(
        [
            ("clip", QuantileClipTransformer()),
            ("scale", RobustScaler()),
            (
                "select",
                SelectKBest(score_func=mutual_info_classif, k=k_best),
            ),
        ]
    ).set_output(transform="default")
