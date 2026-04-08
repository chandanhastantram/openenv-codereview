"""
Core CodeReview OpenEnv environment.

Implements the OpenEnv interface: reset(), step(), state().
"""
from __future__ import annotations

import copy
from typing import Any, Optional

# Score boundaries 
_REWARD_MIN = 0.0
_REWARD_MAX = 1.0


def _clamp_reward(r: float) -> float:
    """Clamp a reward to the closed interval [0.0, 1.0]."""
    return round(max(_REWARD_MIN, min(_REWARD_MAX, float(r))), 4)

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
        self._last_score: float = 0.0

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
        self._last_score = 0.0

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
                reward=0.0,
                done=True,
                info={"error": "Episode already done. Call reset()."},
            )

        self._step += 1
        self._last_error = None
        reward = 0.0

        try:
            if action.action_type == "comment":
                self._handle_comment(action)
                if not self._last_error:
                    new_score = self._compute_current_score()
                    reward = new_score - self._last_score
                    self._last_score = new_score
            elif action.action_type in ("approve", "request_changes"):
                self._done = True
                new_score = self._compute_current_score()
                reward = new_score - self._last_score
                self._last_score = new_score
            else:
                self._last_error = f"Unknown action_type: {action.action_type}"
        except Exception as exc:
            self._last_error = str(exc)

        # Check if episode should end
        if self._step >= self._max_steps and not self._done:
            # Force-submit at max steps
            self._done = True
            new_score = self._compute_current_score()
            reward = new_score - self._last_score
            self._last_score = new_score

        # Ensure incremental reward doesn't push cumulative negatively (though our grader monotonic)
        reward = round(reward, 4)

        self._episode_rewards.append(reward)
        self._cumulative_reward += reward

        return StepResult(
            observation=self._build_observation(),
            reward=reward,
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

    def _handle_comment(self, action: CodeReviewAction) -> None:
        """Place a review comment."""
        if not action.file_path:
            self._last_error = "file_path is required for comment action"
            return 

        if action.line_number is None:
            self._last_error = "line_number is required for comment action"
            return 

        comment = ReviewComment(
            file_path=action.file_path,
            line_number=action.line_number,
            message=action.message,
            severity=action.severity or "major",
            step=self._step,
        )
        self._comments.append(comment)

    def _compute_current_score(self) -> float:
        """Compute the cumulative score for the current state of comments."""
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
