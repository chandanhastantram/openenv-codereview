"""
FastAPI REST server for the CodeReview OpenEnv environment.

Endpoints:
    POST /reset          — Reset env with a task_id, returns observation
    POST /step           — Take an action, returns (obs, reward, done, info)
    GET  /state          — Current environment state
    GET  /tasks          — List available tasks
    GET  /health         — Health check
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .env import CodeReviewEnv
from .models import CodeReviewAction, CodeReviewObservation, StepResult
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

# Single global env instance (stateful, one episode at a time)
env = CodeReviewEnv()


# ── Request/Response Schemas ────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = Field(
        "find-obvious-bug",
        description="Task to load. Options: find-obvious-bug, triage-mixed-pr, security-audit",
    )


class StepRequest(BaseModel):
    action_type: str = Field(..., description="comment | approve | request_changes")
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    message: str = Field(..., description="Comment text or review summary")
    severity: Optional[str] = None


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


class HealthResponse(BaseModel):
    status: str = "ok"
    environment: str = "codereview-env"
    version: str = "1.0.0"


# ── Endpoints ───────────────────────────────────────────────────────

@app.post("/reset")
async def reset_endpoint(body: ResetRequest = ResetRequest()) -> dict:
    """Reset the environment for a new episode."""
    try:
        obs = env.reset(task_id=body.task_id)
        return obs.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
async def step_endpoint(body: StepRequest) -> StepResponse:
    """Take a single step in the environment."""
    action = CodeReviewAction(
        action_type=body.action_type,
        file_path=body.file_path,
        line_number=body.line_number,
        message=body.message,
        severity=body.severity,
    )

    result: StepResult = env.step(action)

    return StepResponse(
        observation=result.observation.model_dump(),
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.get("/state")
async def state_endpoint() -> dict:
    """Get the full current environment state."""
    return env.state()


@app.get("/tasks")
async def tasks_endpoint() -> list[dict]:
    """List all available tasks."""
    return list_tasks()


@app.get("/health")
async def health_endpoint() -> HealthResponse:
    """Health check."""
    return HealthResponse()


@app.get("/")
async def root() -> dict:
    """Root — basic info."""
    return {
        "name": "codereview-env",
        "version": "1.0.0",
        "spec": "openenv",
        "tasks": [t["task_id"] for t in list_tasks()],
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health"],
    }

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
