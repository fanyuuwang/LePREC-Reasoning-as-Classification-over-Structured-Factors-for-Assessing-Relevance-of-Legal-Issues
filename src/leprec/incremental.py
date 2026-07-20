"""Incremental context construction for LEPREC issue generation."""


def fact_prefixes(facts: list[str]) -> list[list[str]]:
    """Return non-empty fact prefixes in their original order."""
    cleaned = [fact.strip() for fact in facts if fact and fact.strip()]
    return [cleaned[:end] for end in range(1, len(cleaned) + 1)]


def deduplicate_issues(issues: list[str]) -> list[str]:
    """Keep the first spelling of each case-insensitive issue."""
    seen: set[str] = set()
    result: list[str] = []
    for issue in issues:
        normalized = issue.casefold()
        if normalized not in seen:
            seen.add(normalized)
            result.append(issue)
    return result
