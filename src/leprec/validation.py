"""Cross-file validation for the canonical public LEPREC dataset."""

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_point_facts(facts: Any, dataset_name: str) -> list[str]:
    if not isinstance(facts, list) or not facts or not all(
        isinstance(fact, str) and fact.strip() for fact in facts
    ):
        raise ValueError(f"{dataset_name} facts must be a non-empty list of strings")
    return facts


def _require_case_identity(case: Any, dataset_name: str) -> None:
    if not isinstance(case, dict):
        raise ValueError(f"Each {dataset_name} record must be an object")
    for field in ("case_id", "case_name"):
        value = case.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{dataset_name} records require a non-empty {field}")


def _lic_labels(lic_cases: Any) -> list[int]:
    if not isinstance(lic_cases, list) or not lic_cases:
        raise ValueError("LIC.json must contain a non-empty JSON array")

    labels: list[int] = []
    for case in lic_cases:
        _require_case_identity(case, "LIC")
        _require_point_facts(case.get("facts"), "LIC")
        issues = case.get("issues")
        if not isinstance(issues, list) or not issues:
            raise ValueError("LIC records require a non-empty issues list")
        for issue in issues:
            if not isinstance(issue, dict) or not isinstance(issue.get("issue"), str):
                raise ValueError("LIC issues require an issue string")
            label = issue.get("label")
            if type(label) is not int or label not in (0, 1):
                raise ValueError("LIC relevance labels must be binary 0 or 1")
            expected_name = "Relevant" if label == 1 else "Irrelevant"
            if issue.get("relevance_label") != expected_name:
                raise ValueError("LIC relevance_label must agree with its binary label")
            labels.append(1 if label == 1 else -1)
    return labels


def _validate_licu(licu_cases: Any) -> tuple[int, int]:
    if not isinstance(licu_cases, list) or not licu_cases:
        raise ValueError("LICU.json must contain a non-empty JSON array")

    issue_count = 0
    for case in licu_cases:
        _require_case_identity(case, "LICU")
        _require_point_facts(case.get("facts"), "LICU")
        issues = case.get("issues")
        if not isinstance(issues, list) or not issues:
            raise ValueError("LICU records require a non-empty issues list")
        for issue in issues:
            if not isinstance(issue, dict) or set(issue) != {
                "issue",
                "relevance_status",
            }:
                raise ValueError("LICU issues must contain only source-extracted provenance")
            if not isinstance(issue["issue"], str) or not issue["issue"].strip():
                raise ValueError("LICU issues require a non-empty issue string")
            if issue["relevance_status"] != "Source-extracted (unlabeled)":
                raise ValueError("LICU issues must be source-extracted and unlabeled")
            issue_count += 1
    return len(licu_cases), issue_count


def _question_metadata(question: Any) -> tuple[str, str, str]:
    if not isinstance(question, dict):
        raise ValueError("Reasoning questions must be objects")
    values = tuple(question.get(field) for field in ("id", "question", "explanation"))
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError("Reasoning questions require non-empty id, question, and explanation")
    return values


def _validate_features(
    features: Any, expected_labels: list[int], questions: list[Any]
) -> None:
    if not isinstance(features, list) or not features:
        raise ValueError("Feature file must contain a non-empty JSON array")
    if len(features) != len(questions):
        raise ValueError("Feature question count must match reasoning-question count")

    for question, feature in zip(questions, features, strict=True):
        if _question_metadata(feature) != _question_metadata(question):
            raise ValueError("Feature metadata must match reasoning-question metadata and order")
        labels = feature.get("attached_labels")
        if labels != expected_labels:
            raise ValueError("Feature labels do not match ordered LIC labels after 0 -> -1 encoding")
        scores = feature.get("attached_scores")
        if not isinstance(scores, list) or len(scores) != len(expected_labels):
            raise ValueError("Feature score rows must match the LIC issue count")
        if not all(
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(score)
            and 0.0 <= score <= 1.0
            for score in scores
        ):
            raise ValueError("Feature scores must be finite probabilities in [0, 1]")


def _validate_manifest(release_root: Path) -> None:
    manifest_path = release_root / "expert_annotations" / "manifest.json"
    manifest = _read_json(manifest_path)
    checksums = manifest.get("sha256") if isinstance(manifest, dict) else None
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError("Manifest must contain non-empty sha256 checksums")
    for relative_path, expected_digest in checksums.items():
        if not isinstance(relative_path, str) or not isinstance(expected_digest, str):
            raise ValueError("Manifest checksum entries must be strings")
        manifest_path = Path(relative_path)
        if manifest_path.is_absolute() or ".." in manifest_path.parts:
            raise ValueError("Manifest paths must remain inside the release root")
        candidate = release_root / manifest_path
        if not candidate.is_file() or _sha256(candidate) != expected_digest:
            raise ValueError(f"Manifest checksum mismatch for {relative_path}")


def validate_release_dataset(
    release_root: str | Path, *, verify_manifest: bool = True
) -> dict[str, int | list[int]]:
    """Validate canonical LIC, LICU, reasoning-question, and feature artifacts."""
    root = Path(release_root)
    lic_cases = _read_json(root / "LIC.json")
    licu_cases = _read_json(root / "LICU.json")
    questions = _read_json(root / "reasoning_questions.json")
    features = _read_gzip_json(root / "expert_annotations" / "phi4_question_scores.json.gz")

    expected_labels = _lic_labels(lic_cases)
    licu_case_count, licu_issue_count = _validate_licu(licu_cases)
    if not isinstance(questions, list) or not questions:
        raise ValueError("reasoning_questions.json must contain a non-empty JSON array")
    question_metadata = [_question_metadata(question) for question in questions]
    if len({metadata[0] for metadata in question_metadata}) != len(question_metadata):
        raise ValueError("Reasoning-question IDs must be unique")
    _validate_features(features, expected_labels, questions)

    if verify_manifest:
        _validate_manifest(root)

    return {
        "licl_case_count": len(lic_cases),
        "licl_issue_count": len(expected_labels),
        "licu_case_count": licu_case_count,
        "licu_issue_count": licu_issue_count,
        "reasoning_question_count": len(questions),
        "feature_shape": [len(expected_labels), len(features)],
    }
