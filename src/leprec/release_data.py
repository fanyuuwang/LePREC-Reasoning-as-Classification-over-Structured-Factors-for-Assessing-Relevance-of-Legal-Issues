"""Deterministically stage LEPREC evaluation data for public release."""

import gzip
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .features import load_question_features
from .validation import validate_release_dataset


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _gzip_copy(source: Path, destination: Path) -> None:
    with source.open("rb") as input_handle, destination.open("wb") as destination_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=destination_handle, mtime=0
        ) as output_handle:
            shutil.copyfileobj(input_handle, output_handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten_labels(labeled_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for case in labeled_cases:
        case_id = case["SID"]
        for issue_id, issue in enumerate(case["issues"]):
            labels.append(
                {
                    "case_id": case_id,
                    "issue_id": issue_id,
                    "issue": issue["issue"],
                    "label": int(issue["label"]),
                }
            )
    return labels


def _case_names(cases: list[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for case in cases:
        case_id = case.get("CaseID")
        case_name = case.get("CaseName")
        if isinstance(case_id, str) and isinstance(case_name, str) and case_name.strip():
            names[case_id] = case_name.strip()
    return names


def _consolidated_lic(
    labeled_cases: list[dict[str, Any]], case_names: dict[str, str]
) -> list[dict[str, Any]]:
    missing_case_names = [case["SID"] for case in labeled_cases if case["SID"] not in case_names]
    if missing_case_names:
        raise ValueError(
            "Missing case names for labelled case IDs: "
            + ", ".join(sorted(missing_case_names))
        )

    dataset: list[dict[str, Any]] = []
    for case in labeled_cases:
        issues = []
        for issue in case["issues"]:
            label = int(issue["label"])
            if label not in (0, 1):
                raise ValueError("LIC relevance labels must be binary 0 or 1")
            issues.append(
                {
                    "issue": issue["issue"],
                    "relevance_label": "Relevant" if label else "Irrelevant",
                    "label": label,
                }
            )
        dataset.append(
            {
                "case_id": case["SID"],
                "case_name": case_names[case["SID"]],
                "facts": case["facts"],
                "issues": issues,
            }
        )
    return dataset


def _consolidated_licu(
    source_cases: list[dict[str, Any]], case_names: dict[str, str]
) -> list[dict[str, Any]]:
    """Normalize source-extracted LICU issue--fact pairs by case.

    LICU is not expert annotated.  Its issue records therefore carry a
    provenance status rather than a binary relevance label.
    """
    missing_case_names = [case["SID"] for case in source_cases if case["SID"] not in case_names]
    if missing_case_names:
        raise ValueError(
            "Missing case names for LICU case IDs: "
            + ", ".join(sorted(missing_case_names))
        )

    dataset: list[dict[str, Any]] = []
    for case in source_cases:
        facts = case["Scenario"]
        issues = case["Issues"]
        if not isinstance(facts, list) or not all(isinstance(fact, str) for fact in facts):
            raise ValueError("LICU facts must be a list of point-form strings")
        if not isinstance(issues, list) or not all(isinstance(issue, str) for issue in issues):
            raise ValueError("LICU issues must be a list of strings")
        dataset.append(
            {
                "case_id": case["SID"],
                "case_name": case_names[case["SID"]],
                "facts": facts,
                "issues": [
                    {
                        "issue": issue,
                        "relevance_status": "Source-extracted (unlabeled)",
                    }
                    for issue in issues
                ],
            }
        )
    return dataset


def prepare_release_dataset(
    source_root: str | Path, release_root: str | Path
) -> dict[str, int | list[int] | dict[str, str]]:
    """Create the public data layout from files used by LEPREC evaluation.

    All inputs are read from ``source_root``. The function writes only beneath
    ``release_root`` and retains the original evaluation JSON unchanged as
    annotation provenance.
    """
    source = Path(source_root)
    destination = Path(release_root)
    cases_dir = destination / "case_texts"
    annotations_dir = destination / "expert_annotations"
    cases_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    labeled_cases = _read_json(source / "final_labeled_data.json")
    if not isinstance(labeled_cases, list):
        raise ValueError("final_labeled_data.json must contain a JSON array")

    cases = [
        {"case_id": case["SID"], "facts": case["facts"]}
        for case in labeled_cases
    ]
    labels = _flatten_labels(labeled_cases)
    _write_json(cases_dir / "licl_cases.json", cases)
    _write_json(annotations_dir / "licl_issue_labels.json", labels)

    iracs_cases = _read_json(source / "iracs_8B.json")
    if not isinstance(iracs_cases, list):
        raise ValueError("iracs_8B.json must contain a JSON array")
    lic_dataset = _consolidated_lic(labeled_cases, _case_names(iracs_cases))
    _write_json(destination / "LIC.json", lic_dataset)

    licu_source_cases = _read_json(source / "total_incremental_issues.json")
    if not isinstance(licu_source_cases, list):
        raise ValueError("total_incremental_issues.json must contain a JSON array")
    licu_dataset = _consolidated_licu(licu_source_cases, _case_names(iracs_cases))
    _write_json(destination / "LICU.json", licu_dataset)

    reasoning_questions = _read_json(source / "reasoning_questions_list.json")
    if not isinstance(reasoning_questions, list):
        raise ValueError("reasoning_questions_list.json must contain a JSON array")
    _write_json(destination / "reasoning_questions.json", reasoning_questions)

    for filename in ("truth_test.json", "truth_test2.json"):
        _write_json(
            annotations_dir / f"evaluation_{filename}",
            _read_json(source / filename),
        )

    feature_source = source / "selected_questions_post_phi.json"
    feature_destination = annotations_dir / "phi4_question_scores.json.gz"
    _gzip_copy(feature_source, feature_destination)
    matrix, feature_labels, feature_questions = load_question_features(feature_destination)
    if len(labels) != len(feature_labels):
        raise ValueError(
            "LICL annotation count does not match the Phi feature-score row count"
        )
    validation_summary = validate_release_dataset(destination, verify_manifest=False)

    output_files = [
        destination / "LIC.json",
        destination / "LICU.json",
        destination / "reasoning_questions.json",
        cases_dir / "licl_cases.json",
        annotations_dir / "licl_issue_labels.json",
        annotations_dir / "evaluation_truth_test.json",
        annotations_dir / "evaluation_truth_test2.json",
        feature_destination,
    ]
    checksums = {
        str(path.relative_to(destination)): _sha256(path) for path in output_files
    }
    report: dict[str, int | list[int] | dict[str, str]] = {
        "case_count": len(cases),
        "label_count": len(labels),
        "licu_case_count": len(licu_dataset),
        "licu_issue_count": sum(len(case["issues"]) for case in licu_dataset),
        "reasoning_question_count": validation_summary["reasoning_question_count"],
        "question_count": len(feature_questions),
        "feature_row_count": int(matrix.shape[0]),
        "feature_column_count": int(matrix.shape[1]),
        "feature_shape": validation_summary["feature_shape"],
        "sha256": checksums,
    }
    _write_json(annotations_dir / "manifest.json", report)
    validate_release_dataset(destination)
    return report
