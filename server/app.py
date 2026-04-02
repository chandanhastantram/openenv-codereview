"""
FastAPI REST server for the CodeReview OpenEnv environment.

Endpoints:
    POST /reset          — Reset env with a task_id, returns (observation, session_id)
    POST /step           — Take an action (requires session_id), returns (obs, reward, done, info)
    GET  /state          — Current environment state (requires session_id query param)
    GET  /tasks          — List available tasks
    GET  /health         — Health check
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .env import CodeReviewEnv
from .models import CodeReviewAction, StepResult
from .tasks import list_tasks

# ── App Setup ───────────────────────────────────────────────────────

app = FastAPI(
    title="CodeReview OpenEnv",
    description="An OpenEnv environment for AI-assisted code review evaluation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session store — one env instance per session ─────────────────────
# Each key: session_id (str) → {"env": CodeReviewEnv, "created_at": float}
_sessions: dict[str, dict] = {}

SESSION_TTL_SECONDS = 60 * 30  # 30 minutes


def _get_or_raise(session_id: str) -> CodeReviewEnv:
    """Retrieve a session's env, raising 404 if not found or expired."""
    record = _sessions.get(session_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call /reset first.",
        )
    # Evict expired sessions lazily
    _evict_expired()
    return record["env"]


def _evict_expired() -> None:
    """Remove sessions older than SESSION_TTL_SECONDS."""
    now = time.time()
    expired = [
        sid for sid, rec in _sessions.items()
        if now - rec["created_at"] > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _sessions[sid]


# ── Request / Response Schemas ──────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = Field(
        "find-obvious-bug",
        description="Task to load. Options: find-obvious-bug, triage-mixed-pr, security-audit",
    )
    session_id: Optional[str] = Field(
        None,
        description="Reuse an existing session (optional). Omit to create a new one.",
    )


class ResetResponse(BaseModel):
    session_id: str
    observation: dict


class StepRequest(BaseModel):
    session_id: str = Field(..., description="Session ID returned by /reset")
    action_type: str = Field(..., description="comment | approve | request_changes")
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    message: str = Field(..., description="Comment text or review summary")
    severity: Optional[str] = None


class StepResponse(BaseModel):
    session_id: str
    observation: dict
    reward: float
    done: bool
    info: dict


class HealthResponse(BaseModel):
    status: str = "ok"
    environment: str = "codereview-env"
    version: str = "1.0.0"
    active_sessions: int = 0


# ── Endpoints ───────────────────────────────────────────────────────

@app.post("/reset", response_model=ResetResponse)
async def reset_endpoint(body: ResetRequest = ResetRequest()) -> ResetResponse:
    """Reset (or create) an isolated session for a new episode."""
    _evict_expired()

    # Reuse or create session
    session_id = body.session_id or str(uuid.uuid4())

    if session_id in _sessions:
        env = _sessions[session_id]["env"]
    else:
        env = CodeReviewEnv()
        _sessions[session_id] = {"env": env, "created_at": time.time()}

    try:
        obs = env.reset(task_id=body.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return ResetResponse(session_id=session_id, observation=obs.model_dump())


@app.post("/step", response_model=StepResponse)
async def step_endpoint(body: StepRequest) -> StepResponse:
    """Take a single step in the environment for the given session."""
    env = _get_or_raise(body.session_id)

    action = CodeReviewAction(
        action_type=body.action_type,
        file_path=body.file_path,
        line_number=body.line_number,
        message=body.message,
        severity=body.severity,
    )

    result: StepResult = env.step(action)

    return StepResponse(
        session_id=body.session_id,
        observation=result.observation.model_dump(),
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.get("/state")
async def state_endpoint(
    session_id: str = Query(..., description="Session ID from /reset")
) -> dict:
    """Get the full current environment state for a session."""
    env = _get_or_raise(session_id)
    return env.state()


@app.get("/tasks")
async def tasks_endpoint() -> list[dict]:
    """List all available tasks."""
    return list_tasks()


@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """Health check — also reports active session count."""
    _evict_expired()
    return HealthResponse(active_sessions=len(_sessions))


@app.get("/")
async def root() -> dict:
    """Root — basic info."""
    return {
        "name": "codereview-env",
        "version": "1.0.0",
        "spec": "openenv",
        "tasks": [t["task_id"] for t in list_tasks()],
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health"],
        "note": "All /step and /state calls require a session_id from /reset.",
    }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
