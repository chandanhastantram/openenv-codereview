"""
Core CodeReview OpenEnv environment.

Implements the OpenEnv interface: reset(), step(), state().
"""
from __future__ import annotations

import copy
from typing import Any, Optional

from .models import (
    CodeReviewAction,
    CodeReviewObservation,
    CodeReviewReward,
    DiffHunk,
    EnvState,
    FileChange,
    ReviewComment,
    StepResult,
)
from .tasks import grade_task, load_task, list_tasks


MAX_STEPS = 10


class CodeReviewEnv:
    """
    OpenEnv-compliant code review environment.

    An agent reviews a pull request by:
    1. Reading the PR diff (observation)
    2. Placing comments on specific lines (action)
    3. Submitting a final review (approve / request_changes)

    Reward is computed by a deterministic grader at episode end.
    """

    def __init__(self):
        self._task_id: str = ""
        self._task_data: dict = {}
        self._ground_truth: list[dict] = []
        self._step: int = 0
        self._max_steps: int = MAX_STEPS
        self._done: bool = True
        self._comments: list[ReviewComment] = []
        self._cumulative_reward: float = 0.0
        self._episode_rewards: list[float] = []
        self._last_error: Optional[str] = None

    # ── OpenEnv Interface ───────────────────────────────────────────

    def reset(self, task_id: str = "find-obvious-bug") -> CodeReviewObservation:
        """
        Reset the environment for a new episode.

        Args:
            task_id: Which task to load. One of:
                     'find-obvious-bug', 'triage-mixed-pr', 'security-audit'

        Returns:
            Initial observation with the PR diff.
        """
        self._task_data = load_task(task_id)
        self._task_id = task_id
        self._ground_truth = self._task_data["ground_truth"]
        self._step = 0
        self._done = False
        self._comments = []
        self._cumulative_reward = 0.0
        self._episode_rewards = []
        self._last_error = None

        return self._build_observation()

    def step(self, action: CodeReviewAction) -> StepResult:
        """
        Execute an agent action and return the result.

        Actions:
        - 'comment': Place a review comment on a file/line
        - 'approve': Submit the review as approved (ends episode)
        - 'request_changes': Submit with changes requested (ends episode)

        Returns:
            StepResult with observation, reward, done, info
        """
        if self._done:
            return StepResult(
                observation=self._build_observation(),
                reward=0.01,  # Must be > 0 per OpenEnv validator (survives :.2f)
                done=True,
                info={"error": "Episode already done. Call reset()."},
            )

        self._step += 1
        self._last_error = None
        reward = 0.01  # Default: always > 0 (survives :.2f formatting)

        try:
            if action.action_type == "comment":
                reward = self._handle_comment(action)
            elif action.action_type in ("approve", "request_changes"):
                reward = self._handle_submit(action)
            else:
                self._last_error = f"Unknown action_type: {action.action_type}"
                # reward stays at 0.01 (valid, > 0)
        except Exception as exc:
            self._last_error = str(exc)

        # Check if episode should end
        if self._step >= self._max_steps and not self._done:
            # Force-submit at max steps
            reward = self._finalize_episode()

        self._episode_rewards.append(reward)
        self._cumulative_reward += reward

        return StepResult(
            observation=self._build_observation(),
            reward=round(reward, 4),
            done=self._done,
            info={
                "step": self._step,
                "cumulative_reward": round(self._cumulative_reward, 4),
                "comments_placed": len(self._comments),
            },
        )

    def state(self) -> dict:
        """Return the full serializable environment state."""
        return EnvState(
            task_id=self._task_id,
            step=self._step,
            max_steps=self._max_steps,
            done=self._done,
            comments=copy.deepcopy(self._comments),
            cumulative_reward=round(self._cumulative_reward, 4),
            episode_rewards=[round(r, 4) for r in self._episode_rewards],
        ).model_dump()

    # ── Action handlers ─────────────────────────────────────────────

    def _handle_comment(self, action: CodeReviewAction) -> float:
        """Place a review comment. Returns incremental reward signal in (0, 1)."""
        if not action.file_path:
            self._last_error = "file_path is required for comment action"
            # Must be > 0 per OpenEnv validator — use minimum valid score
            return 0.01

        if action.line_number is None:
            self._last_error = "line_number is required for comment action"
            return 0.01

        comment = ReviewComment(
            file_path=action.file_path,
            line_number=action.line_number,
            message=action.message,
            severity=action.severity or "major",
            step=self._step,
        )
        self._comments.append(comment)

        # Provide a small incremental signal: did this comment match any issue?
        from .tasks import _match_comment_to_issue

        best = 0.0
        for issue in self._ground_truth:
            matched, score = _match_comment_to_issue(comment, issue)
            if matched:
                best = max(best, score)

        # Scale incremental reward (small, so final grading dominates)
        # Always clamp to (0, 1) — values must survive :.2f formatting
        raw = round(best * 0.05, 4)
        return max(0.01, min(0.99, raw)) if raw > 0 else 0.01

    def _handle_submit(self, action: CodeReviewAction) -> float:
        """Submit the review — finalize and grade."""
        return self._finalize_episode()

    def _finalize_episode(self) -> float:
        """Run the grader and end the episode."""
        self._done = True
        result = grade_task(self._task_id, self._comments, self._ground_truth)
        return result["score"]

    # ── Observation builder ─────────────────────────────────────────

    def _build_observation(self) -> CodeReviewObservation:
        """Build the current observation from internal state."""
        files = []
        for f in self._task_data.get("files", []):
            hunks = [
                DiffHunk(
                    start_line=h["start_line"],
                    end_line=h["end_line"],
                    content=h["content"],
                )
                for h in f.get("hunks", [])
            ]
            files.append(
                FileChange(
                    path=f["path"],
                    language=f.get("language", "python"),
                    change_type=f.get("change_type", "modified"),
                    hunks=hunks,
                    full_new_content=f.get("full_new_content", ""),
                )
            )

        return CodeReviewObservation(
            task_id=self._task_id,
            pr_title=self._task_data.get("pr_title", ""),
            pr_description=self._task_data.get("pr_description", ""),
            files=files,
            existing_comments=copy.deepcopy(self._comments),
            step=self._step,
            max_steps=self._max_steps,
            last_action_error=self._last_error,
            done=self._done,
        )
