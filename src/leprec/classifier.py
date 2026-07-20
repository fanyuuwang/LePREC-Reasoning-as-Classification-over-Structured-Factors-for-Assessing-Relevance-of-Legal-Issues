"""Correlation-aware symbolic prediction for LEPREC."""

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression


class LePRECClassifier:
    """Interpretable standard linear classifier over verifier probabilities."""

    def __init__(self, c: float = 1.0, random_state: int = 42) -> None:
        self.model = LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        )

    def fit(self, features: Sequence[Sequence[float]], labels: Sequence[int]):
        self.model.fit(features, labels)
        return self

    def predict(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        return self.model.predict(features)

    def predict_proba(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        return self.model.predict_proba(features)

    def contributions(self, questions: Sequence[str]) -> dict[str, float]:
        if len(questions) != self.model.coef_.shape[1]:
            raise ValueError("Question count must match fitted feature count")
        return {
            question: float(weight)
            for question, weight in zip(questions, self.model.coef_[0], strict=True)
        }
