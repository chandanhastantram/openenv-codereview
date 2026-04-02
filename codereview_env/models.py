"""
Typed Pydantic models for the CodeReview OpenEnv environment.

Defines the Observation, Action, and Reward schemas that form
the contract between agent and environment.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Sub-models ──────────────────────────────────────────────────────

class DiffHunk(BaseModel):
    """A single hunk of a unified diff."""
    start_line: int = Field(..., description="Start line in the new file")
    end_line: int = Field(..., description="End line in the new file")
    content: str = Field(..., description="Unified diff text for this hunk")


class FileChange(BaseModel):
    """One file changed in the pull request."""
    path: str = Field(..., description="File path, e.g. 'src/utils.py'")
    language: str = Field("python", description="Programming language")
    change_type: Literal["added", "modified", "deleted"] = "modified"
    hunks: list[DiffHunk] = Field(default_factory=list)
    full_new_content: str = Field("", description="Complete new file content for context")


class ReviewComment(BaseModel):
    """A review comment placed by the agent."""
    file_path: str
    line_number: int
    message: str
    severity: Literal["critical", "major", "minor", "suggestion"] = "major"
    step: int = Field(0, description="Step at which this comment was made")


# ── Core OpenEnv Models ─────────────────────────────────────────────

class CodeReviewObservation(BaseModel):
    """What the agent sees each step (the environment state)."""
    task_id: str = Field(..., description="Active task identifier")
    pr_title: str = Field(..., description="Pull request title")
    pr_description: str = Field("", description="Pull request description/body")
    files: list[FileChange] = Field(default_factory=list, description="Changed files with diffs")
    existing_comments: list[ReviewComment] = Field(
        default_factory=list, description="Comments the agent has placed so far"
    )
    step: int = Field(0, description="Current step number (1-indexed)")
    max_steps: int = Field(10, description="Maximum steps in this episode")
    last_action_error: Optional[str] = Field(
        None, description="Error message from the previous action, if any"
    )
    done: bool = Field(False, description="Whether the episode has ended")


class CodeReviewAction(BaseModel):
    """What the agent can do — comment, approve, or request changes."""
    action_type: Literal["comment", "approve", "request_changes"] = Field(
        ..., description="Type of review action"
    )
    file_path: Optional[str] = Field(
        None, description="Target file for a comment (required if action_type='comment')"
    )
    line_number: Optional[int] = Field(
        None, description="Target line for a comment (required if action_type='comment')"
    )
    message: str = Field(
        ..., description="Comment body or review summary"
    )
    severity: Optional[Literal["critical", "major", "minor", "suggestion"]] = Field(
        None, description="Issue severity (for comments)"
    )


class CodeReviewReward(BaseModel):
    """Reward signal returned after each step or at episode end."""
    score: float = Field(0.0, ge=0.0, le=1.0, description="Overall score 0.0–1.0")
    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Per-issue scoring breakdown"
    )
    feedback: str = Field("", description="Human-readable grading explanation")


class StepResult(BaseModel):
    """Full return from env.step()."""
    observation: CodeReviewObservation
    reward: float = 0.0
    done: bool = False
    info: dict = Field(default_factory=dict)


class EnvState(BaseModel):
    """Serializable full environment state for state() endpoint."""
    task_id: str = ""
    step: int = 0
    max_steps: int = 10
    done: bool = False
    comments: list[ReviewComment] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    episode_rewards: list[float] = Field(default_factory=list)
