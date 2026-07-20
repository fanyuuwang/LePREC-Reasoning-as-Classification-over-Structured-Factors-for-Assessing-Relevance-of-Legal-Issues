"""Prompt and response helpers for LEPREC's question-generation stage."""

import json


def build_incremental_prompt(facts: list[str]) -> str:
    """Build the paper's constrained incremental issue-generation prompt."""
    joined_facts = "\n".join(
        f"{index}. {fact}" for index, fact in enumerate(facts, start=1)
    )
    return (
        "Given only these scenario facts, identify the most relevant legal "
        "issues. Do not alter or deviate from their meaning. Format each issue "
        "as 'Whether ...'. Return strictly a JSON array of strings.\n\n"
        f"Facts:\n{joined_facts}"
    )


def parse_issue_list(response: str) -> list[str]:
    """Parse a model response and retain only well-formed legal issues."""
    values = json.loads(response)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("Expected a JSON array of issue strings")
    return [value.strip() for value in values if value.strip().startswith("Whether")]
