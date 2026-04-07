"""
Unit tests for the CodeReview OpenEnv environment.
"""
import sys
import os
import pytest

# Ensure the parent directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.env import CodeReviewEnv
from server.models import CodeReviewAction, ReviewComment
from server.tasks import grade_task, load_task, list_tasks


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return CodeReviewEnv()


# ── Task loading tests ──────────────────────────────────────────────

class TestTaskLoading:
    def test_list_tasks_returns_three(self):
        tasks = list_tasks()
        assert len(tasks) == 3
        ids = [t["task_id"] for t in tasks]
        assert "find-obvious-bug" in ids
        assert "triage-mixed-pr" in ids
        assert "security-audit" in ids

    def test_load_easy_task(self):
        data = load_task("find-obvious-bug")
        assert data["task_id"] == "find-obvious-bug"
        assert len(data["files"]) >= 1
        assert len(data["ground_truth"]) == 1

    def test_load_medium_task(self):
        data = load_task("triage-mixed-pr")
        assert data["task_id"] == "triage-mixed-pr"
        assert len(data["ground_truth"]) == 3

    def test_load_hard_task(self):
        data = load_task("security-audit")
        assert data["task_id"] == "security-audit"
        assert len(data["ground_truth"]) == 3

    def test_load_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            load_task("nonexistent-task")


# ── Environment reset tests ────────────────────────────────────────

class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset("find-obvious-bug")
        assert obs.task_id == "find-obvious-bug"
        assert obs.step == 0
        assert obs.done is False
        assert len(obs.files) >= 1
        assert obs.pr_title != ""

    def test_reset_clears_state(self, env):
        env.reset("find-obvious-bug")
        # Add a comment
        env.step(CodeReviewAction(
            action_type="comment",
            file_path="src/users/profile.py",
            line_number=18,
            message="Bug here",
        ))
        # Reset should clear
        obs = env.reset("find-obvious-bug")
        assert obs.step == 0
        assert len(obs.existing_comments) == 0

    def test_reset_all_tasks(self, env):
        for task_id in ["find-obvious-bug", "triage-mixed-pr", "security-audit"]:
            obs = env.reset(task_id)
            assert obs.task_id == task_id
            assert obs.done is False


# ── Step tests ──────────────────────────────────────────────────────

class TestStep:
    def test_step_comment_increments_step(self, env):
        env.reset("find-obvious-bug")
        result = env.step(CodeReviewAction(
            action_type="comment",
            file_path="src/users/profile.py",
            line_number=18,
            message="user['bio'] can be None, .strip() will raise AttributeError",
            severity="critical",
        ))
        assert result.observation.step == 1
        assert result.done is False
        assert len(result.observation.existing_comments) == 1

    def test_step_approve_ends_episode(self, env):
        env.reset("find-obvious-bug")
        result = env.step(CodeReviewAction(
            action_type="approve",
            message="LGTM",
        ))
        assert result.done is True

    def test_step_request_changes_ends_episode(self, env):
        env.reset("find-obvious-bug")
        # First add a correct comment
        env.step(CodeReviewAction(
            action_type="comment",
            file_path="src/users/profile.py",
            line_number=18,
            message="bio can be None, .strip() will crash with AttributeError",
            severity="critical",
        ))
        # Then submit
        result = env.step(CodeReviewAction(
            action_type="request_changes",
            message="Found a null dereference bug",
        ))
        assert result.done is True
        assert result.reward > 0.0  # Should get some score

    def test_step_after_done_returns_done(self, env):
        env.reset("find-obvious-bug")
        env.step(CodeReviewAction(action_type="approve", message="ok"))
        # Step again after episode ended
        result = env.step(CodeReviewAction(action_type="approve", message="ok"))
        assert result.done is True
        assert result.reward == 0.01  # Clamped: must be > 0 per OpenEnv validator

    def test_step_missing_file_path_for_comment(self, env):
        env.reset("find-obvious-bug")
        result = env.step(CodeReviewAction(
            action_type="comment",
            message="Bug",
        ))
        assert result.observation.last_action_error is not None

    def test_max_steps_auto_submits(self, env):
        env.reset("find-obvious-bug")
        env._max_steps = 2  # Override for testing
        env.step(CodeReviewAction(
            action_type="comment",
            file_path="test.py",
            line_number=1,
            message="comment 1",
        ))
        result = env.step(CodeReviewAction(
            action_type="comment",
            file_path="test.py",
            line_number=2,
            message="comment 2",
        ))
        assert result.done is True


# ── State tests ─────────────────────────────────────────────────────

class TestState:
    def test_state_returns_dict(self, env):
        env.reset("find-obvious-bug")
        state = env.state()
        assert isinstance(state, dict)
        assert state["task_id"] == "find-obvious-bug"
        assert state["step"] == 0
        assert state["done"] is False


# ── Grader tests ────────────────────────────────────────────────────

class TestGraders:
    def test_easy_grader_perfect_match(self):
        data = load_task("find-obvious-bug")
        comments = [
            ReviewComment(
                file_path="src/users/profile.py",
                line_number=18,
                message="user['bio'] can be None — calling .strip() on None raises AttributeError",
                severity="critical",
            )
        ]
        result = grade_task("find-obvious-bug", comments, data["ground_truth"])
        assert result["score"] >= 0.99  # Clamped from 1.0 to _SCORE_MAX

    def test_easy_grader_wrong_line(self):
        data = load_task("find-obvious-bug")
        comments = [
            ReviewComment(
                file_path="src/users/profile.py",
                line_number=10,
                message="bio might be None, strip could fail",
                severity="critical",
            )
        ]
        result = grade_task("find-obvious-bug", comments, data["ground_truth"])
        assert 0.3 <= result["score"] <= 0.6

    def test_easy_grader_no_match(self):
        data = load_task("find-obvious-bug")
        comments = [
            ReviewComment(
                file_path="src/users/profile.py",
                line_number=5,
                message="This function looks good",
                severity="suggestion",
            )
        ]
        result = grade_task("find-obvious-bug", comments, data["ground_truth"])
        assert result["score"] <= 0.01  # Clamped from 0.0 to _SCORE_MIN

    def test_medium_grader_all_found(self):
        data = load_task("triage-mixed-pr")
        comments = [
            ReviewComment(
                file_path="src/orders/processor.py",
                line_number=44,
                message="Race condition: self.results is shared and not thread-safe",
                severity="critical",
            ),
            ReviewComment(
                file_path="src/orders/validation.py",
                line_number=19,
                message="Should check quantity <= 0, not just < 0. Zero quantity is invalid.",
                severity="major",
            ),
            ReviewComment(
                file_path="src/orders/processor.py",
                line_number=3,
                message="Unused import: Any is imported but never used",
                severity="minor",
            ),
        ]
        result = grade_task("triage-mixed-pr", comments, data["ground_truth"])
        assert result["score"] >= 0.8

    def test_hard_grader_all_vulns_found(self):
        data = load_task("security-audit")
        comments = [
            ReviewComment(
                file_path="src/api/admin_search.py",
                line_number=25,
                message="SQL injection: user input is interpolated into query via f-strings",
                severity="critical",
            ),
            ReviewComment(
                file_path="src/api/admin_search.py",
                line_number=12,
                message="Hardcoded password and secret. Use environment variables.",
                severity="critical",
            ),
            ReviewComment(
                file_path="src/api/admin_search.py",
                line_number=42,
                message="No authentication or authorization on admin endpoints",
                severity="critical",
            ),
        ]
        result = grade_task("security-audit", comments, data["ground_truth"])
        assert result["score"] >= 0.9

    def test_hard_grader_nothing_found(self):
        data = load_task("security-audit")
        comments = [
            ReviewComment(
                file_path="src/api/admin_search.py",
                line_number=1,
                message="Looks fine to me",
                severity="suggestion",
            )
        ]
        result = grade_task("security-audit", comments, data["ground_truth"])
        assert result["score"] <= 0.01  # Clamped from 0.0 to _SCORE_MIN

    def test_rewards_strictly_in_open_interval(self):
        """All graders must return scores strictly in (0, 1) — never 0.0 or 1.0."""
        for task_id in ["find-obvious-bug", "triage-mixed-pr", "security-audit"]:
            data = load_task(task_id)
            # Empty comments — would be 0.0 raw, must be clamped above 0
            r = grade_task(task_id, [], data["ground_truth"])
            assert 0.0 < r["score"] < 1.0, f"{task_id} empty: {r['score']} not in (0,1)"
            # Random wrong comment — should also be clamped
            r = grade_task(task_id, [
                ReviewComment(file_path="x.py", line_number=999, message="nothing", severity="minor")
            ], data["ground_truth"])
            assert 0.0 < r["score"] < 1.0, f"{task_id} wrong: {r['score']} not in (0,1)"
