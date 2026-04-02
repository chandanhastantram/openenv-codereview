"""
Task definitions and deterministic graders for CodeReview OpenEnv.

Each task has:
- A PR loaded from JSON data
- A ground_truth list of expected issues
- A grader that compares agent comments against ground truth

Graders are fully deterministic — they use keyword matching + line-number
proximity to score agent comments against known issues.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import ReviewComment

DATA_DIR = Path(__file__).parent / "data"

# ── Line proximity tolerance ────────────────────────────────────────

LINE_TOLERANCE = 5  # Agent's line must be within ±5 of ground truth


# ── Task data loader ────────────────────────────────────────────────

def load_task(task_id: str) -> dict:
    """Load task data from JSON."""
    filename_map = {
        "find-obvious-bug": "easy_pr.json",
        "triage-mixed-pr": "medium_pr.json",
        "security-audit": "hard_pr.json",
    }
    filename = filename_map.get(task_id)
    if not filename:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(filename_map.keys())}")

    path = DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_tasks() -> list[dict[str, str]]:
    """Return metadata for all available tasks."""
    return [
        {
            "task_id": "find-obvious-bug",
            "difficulty": "easy",
            "description": "Find an obvious null-dereference bug in a user profile utility.",
        },
        {
            "task_id": "triage-mixed-pr",
            "difficulty": "medium",
            "description": "Triage a PR with 3 issues: a race condition (critical), missing validation (major), and unused import (minor).",
        },
        {
            "task_id": "security-audit",
            "difficulty": "hard",
            "description": "Audit an admin endpoint for SQL injection, hardcoded secrets, and missing authentication.",
        },
    ]


# ── Keyword matching ────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _keywords_match(comment_text: str, keywords: list[str]) -> bool:
    """Check if the agent's comment mentions any of the expected keywords."""
    normalized = _normalize(comment_text)
    for kw in keywords:
        if _normalize(kw) in normalized:
            return True
    return False


def _line_matches(agent_line: int | None, truth_line: int) -> bool:
    """Check if the agent's line number is close enough to ground truth."""
    if agent_line is None:
        return False
    return abs(agent_line - truth_line) <= LINE_TOLERANCE


def _file_matches(agent_path: str | None, truth_path: str) -> bool:
    """Check if agent targeted the correct file (flexible basename matching)."""
    if not agent_path:
        return False
    # Allow both "src/users/profile.py" or just "profile.py"
    return (
        agent_path == truth_path
        or agent_path.endswith("/" + truth_path.split("/")[-1])
        or truth_path.endswith("/" + agent_path.split("/")[-1])
        or agent_path.split("/")[-1] == truth_path.split("/")[-1]
    )


# ── Issue matching ──────────────────────────────────────────────────

def _match_comment_to_issue(
    comment: ReviewComment, issue: dict
) -> tuple[bool, float]:
    """
    Check if a comment matches a ground-truth issue.
    Returns (matched, score_contribution).

    Scoring:
    - Correct file + correct line + keyword match → 1.0
    - Correct file + keyword match (wrong line)   → 0.6
    - Keyword match anywhere                      → 0.3
    """
    kw_match = _keywords_match(comment.message, issue["keywords"])
    file_match = _file_matches(comment.file_path, issue["file_path"])
    line_match = _line_matches(comment.line_number, issue["line_number"])

    if file_match and line_match and kw_match:
        return True, 1.0
    elif file_match and kw_match:
        return True, 0.6
    elif kw_match:
        return True, 0.3

    return False, 0.0


# ── Graders ─────────────────────────────────────────────────────────

def grade_easy(comments: list[ReviewComment], ground_truth: list[dict]) -> dict:
    """
    Grade the easy task: 1 issue, binary scoring with partial credit.

    Returns: { score, breakdown, feedback }
    """
    issue = ground_truth[0]
    best_score = 0.0

    for comment in comments:
        matched, score = _match_comment_to_issue(comment, issue)
        if matched:
            best_score = max(best_score, score)

    feedback_parts = []
    if best_score >= 1.0:
        feedback_parts.append(f"✅ Found the bug with correct file and line.")
    elif best_score >= 0.6:
        feedback_parts.append(f"⚠️ Found the bug but pointed to wrong line.")
    elif best_score > 0:
        feedback_parts.append(f"⚠️ Mentioned the issue but didn't pinpoint location.")
    else:
        feedback_parts.append(f"❌ Did not identify the null-dereference bug.")

    return {
        "score": min(best_score, 1.0),
        "breakdown": {issue["id"]: best_score},
        "feedback": " ".join(feedback_parts),
    }


def grade_medium(comments: list[ReviewComment], ground_truth: list[dict]) -> dict:
    """
    Grade the medium task: 3 issues with severity weighting.

    Weights: critical=0.50, major=0.30, minor=0.20
    """
    severity_weights = {"critical": 0.50, "major": 0.30, "minor": 0.20}
    breakdown: dict[str, float] = {}
    total_score = 0.0
    feedback_parts = []

    for issue in ground_truth:
        weight = severity_weights.get(issue["severity"], 0.25)
        best_score = 0.0

        for comment in comments:
            matched, score = _match_comment_to_issue(comment, issue)
            if matched:
                best_score = max(best_score, score)

        weighted_score = best_score * weight
        breakdown[issue["id"]] = round(weighted_score, 3)
        total_score += weighted_score

        if best_score >= 0.6:
            feedback_parts.append(f"✅ {issue['id']}: found ({best_score:.0%} match)")
        else:
            feedback_parts.append(f"❌ {issue['id']}: missed")

    return {
        "score": round(min(total_score, 1.0), 3),
        "breakdown": breakdown,
        "feedback": " | ".join(feedback_parts),
    }


def grade_hard(comments: list[ReviewComment], ground_truth: list[dict]) -> dict:
    """
    Grade the hard task: 3 security vulnerabilities.

    Base: 0.30 per vuln found. Bonus 0.033 per correct severity tag.
    Max 1.0.
    """
    per_issue = 0.30
    severity_bonus = 0.033
    breakdown: dict[str, float] = {}
    total_score = 0.0
    feedback_parts = []

    for issue in ground_truth:
        best_score = 0.0
        best_comment: ReviewComment | None = None

        for comment in comments:
            matched, score = _match_comment_to_issue(comment, issue)
            if matched and score > best_score:
                best_score = score
                best_comment = comment

        issue_score = 0.0
        if best_score > 0:
            issue_score = per_issue * best_score
            # Severity bonus
            if (
                best_comment
                and best_comment.severity
                and best_comment.severity.lower() == issue["severity"].lower()
            ):
                issue_score += severity_bonus
                feedback_parts.append(
                    f"✅ {issue['id']}: found + correct severity"
                )
            else:
                feedback_parts.append(f"⚠️ {issue['id']}: found, wrong severity")
        else:
            feedback_parts.append(f"❌ {issue['id']}: missed")

        breakdown[issue["id"]] = round(issue_score, 3)
        total_score += issue_score

    return {
        "score": round(min(total_score, 1.0), 3),
        "breakdown": breakdown,
        "feedback": " | ".join(feedback_parts),
    }


# ── Grader dispatch ─────────────────────────────────────────────────

GRADERS = {
    "find-obvious-bug": grade_easy,
    "triage-mixed-pr": grade_medium,
    "security-audit": grade_hard,
}


def grade_task(
    task_id: str,
    comments: list[ReviewComment],
    ground_truth: list[dict],
) -> dict:
    """Run the appropriate grader for a task."""
    grader = GRADERS.get(task_id)
    if not grader:
        raise ValueError(f"No grader for task: {task_id}")
    return grader(comments, ground_truth)
