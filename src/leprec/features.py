"""Probability-feature loading for LEPREC's symbolic classifier."""

import gzip
import json
from pathlib import Path

import numpy as np


def load_question_features(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load question-major scores and return sample-major feature rows.

    The release matrix stores one object per reasoning question. Each object
    carries scores for all labelled issue-fact pairs, so the score matrix is
    transposed before fitting the symbolic classifier.
    """
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        questions = json.load(handle)

    if not isinstance(questions, list) or not questions:
        raise ValueError("Question feature file must be a non-empty JSON array")

    first = questions[0]
    if not isinstance(first, dict) or "attached_labels" not in first:
        raise ValueError("Question feature file is missing attached_labels")

    labels = first["attached_labels"]
    if not isinstance(labels, list) or not labels:
        raise ValueError("attached_labels must be a non-empty list")

    scores: list[list[float]] = []
    names: list[str] = []
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Each question feature must be an object")
        if question.get("attached_labels") != labels:
            raise ValueError("Each question feature must use the same attached_labels")
        score_row = question.get("attached_scores")
        if not isinstance(score_row, list) or len(score_row) != len(labels):
            raise ValueError("Each attached_scores row must match attached_labels")
        scores.append(score_row)
        names.append(str(question.get("question", "")))

    return (
        np.asarray(scores, dtype=float).T,
        np.asarray(labels, dtype=int),
        names,
    )
