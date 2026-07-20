"""Nested cross-validation for LEPREC's standard linear alternatives."""

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC


def _model_specs(seed: int) -> dict[str, tuple[object, dict[str, list[float]]]]:
    values = [0.01, 0.1, 1.0, 10.0, 100.0]
    return {
        "logistic_regression": (
            LogisticRegression(
                class_weight="balanced", max_iter=5000, random_state=seed
            ),
            {"C": values},
        ),
        "linear_svc": (
            SVC(kernel="linear", class_weight="balanced", random_state=seed),
            {"C": values},
        ),
        "ridge": (
            RidgeClassifier(class_weight="balanced", random_state=seed),
            {"alpha": values},
        ),
    }


def _metric_summary(folds: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": float(np.mean([fold[metric] for fold in folds])),
            "std": float(np.std([fold[metric] for fold in folds])),
        }
        for metric in folds[0]
    }


def evaluate_linear_models(
    features: np.ndarray,
    labels: np.ndarray,
    folds: int = 5,
    seed: int = 42,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate paper-aligned linear models with nested stratified CV.

    Grid search occurs inside every outer training fold, avoiding use of test
    examples when selecting regularization strength.
    """
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("Features must be a two-dimensional matrix aligned with labels")
    if len(np.unique(y)) != 2:
        raise ValueError("LEPREC evaluation requires exactly two relevance labels")
    if min(np.bincount(np.unique(y, return_inverse=True)[1])) < folds:
        raise ValueError("Each relevance label needs at least one sample per fold")

    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results: dict[str, list[dict[str, float]]] = {
        name: [] for name in _model_specs(seed)
    }

    for train_indices, test_indices in outer.split(X, y):
        for name, (estimator, parameters) in _model_specs(seed).items():
            inner = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            search = GridSearchCV(
                estimator,
                parameters,
                scoring="f1_macro",
                cv=inner,
            )
            search.fit(X[train_indices], y[train_indices])
            predicted = search.predict(X[test_indices])
            precision, recall, f1, _ = precision_recall_fscore_support(
                y[test_indices], predicted, average="macro", zero_division=0
            )
            results[name].append(
                {
                    "accuracy": float(accuracy_score(y[test_indices], predicted)),
                    "precision_macro": float(precision),
                    "recall_macro": float(recall),
                    "f1_macro": float(f1),
                }
            )

    return {name: _metric_summary(model_folds) for name, model_folds in results.items()}
