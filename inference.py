"""
Inference Script — CodeReview OpenEnv
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after episode ends, always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
"""
from __future__ import annotations

import json
import os
import re
import textwrap
import uuid
from typing import List, Optional

import httpx
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL = os.getenv("ENV_URL") or "http://localhost:7860"
BENCHMARK = "codereview-env"

MAX_STEPS = 8
TEMPERATURE = 0.2
MAX_TOKENS = 800

# Score threshold to consider an episode "successful"
SUCCESS_THRESHOLD = 0.5

# Strict bounds for logged scores
_LOG_MIN = 0.0
_LOG_MAX = 1.0


def _clamp(v: float) -> float:
    """Clamp a numeric value to the closed interval [0.0, 1.0]."""
    return max(_LOG_MIN, min(_LOG_MAX, float(v)))


# ── Logging (mandatory format) ─────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    clamped = _clamp(reward)
    print(
        f"[STEP] step={step} action={action} reward={clamped:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    clamped_rewards = [_clamp(r) for r in rewards]
    rewards_str = ",".join(f"{r:.2f}" for r in clamped_rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# ── Environment client (HTTP) ──────────────────────────────────────

class EnvClient:
    """REST client to interact with the CodeReview OpenEnv server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)
        self.session_id: Optional[str] = None

    def reset(self, task_id: str) -> dict:
        resp = self.client.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
        )
        resp.raise_for_status()
        data = resp.json()
        # Store session_id from server (session-isolated env)
        self.session_id = data.get("session_id") or str(uuid.uuid4())
        # Return just the observation (backwards compat)
        return data.get("observation", data)

    def step(self, action: dict) -> dict:
        payload = dict(action)
        payload["session_id"] = self.session_id or ""
        resp = self.client.post(f"{self.base_url}/step", json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_tasks(self) -> list:
        resp = self.client.get(f"{self.base_url}/tasks")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self.client.close()


# ── Prompt engineering ──────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert code reviewer. You are reviewing a pull request.
    
    Your job is to:
    1. Read the PR diff carefully
    2. Identify bugs, security issues, and code quality problems
    3. Place specific comments on the files/lines where you find issues
    4. When done reviewing, submit your review
    
    For EACH issue you find, respond with a JSON object on its own line:
    {"action_type": "comment", "file_path": "<path>", "line_number": <n>, "message": "<description of issue>", "severity": "<critical|major|minor|suggestion>"}
    
    When you are done finding issues and ready to submit, respond with:
    {"action_type": "request_changes", "message": "<summary of all issues found>"}
    
    If you find no issues, respond with:
    {"action_type": "approve", "message": "LGTM - no issues found"}
    
    IMPORTANT: Respond with ONLY ONE JSON object per response. No explanations, no markdown.
""")


def build_initial_prompt(observation: dict) -> str:
    """Build the first user message from the initial observation."""
    task_id = observation.get("task_id", "?")
    title = observation.get("pr_title", "?")
    desc = observation.get("pr_description", "")
    max_steps = observation.get("max_steps", 10)

    # Format files/diffs
    files_text = ""
    for f in observation.get("files", []):
        files_text += f"\n--- File: {f['path']} ({f['change_type']}) ---\n"
        files_text += f["full_new_content"]
        files_text += "\n"

    prompt = textwrap.dedent(f"""\
        Task: {task_id}
        Max steps allowed: {max_steps}
        PR Title: {title}
        PR Description: {desc}
        
        Files changed:
        {files_text}
        
        Review this PR carefully. Find issues one at a time. Respond with a single JSON object.
    """)
    return prompt.strip()


def build_followup_prompt(observation: dict, last_reward: float) -> str:
    """Build a follow-up message after each step, updating the agent on progress."""
    step = observation.get("step", 0)
    max_steps = observation.get("max_steps", 10)
    error = observation.get("last_action_error")
    existing = observation.get("existing_comments", [])

    parts = [f"Step {step}/{max_steps} — your last action earned reward {last_reward:+.3f}."]

    if error:
        parts.append(f"⚠️ Error from last action: {error}")

    if existing:
        parts.append(f"\nComments placed so far ({len(existing)}):")
        for c in existing:
            parts.append(
                f"  [{c.get('severity','?')}] {c.get('file_path','?')}:"
                f"{c.get('line_number','?')} — {c.get('message','')[:80]}"
            )

    remaining = max_steps - step
    if remaining <= 2:
        parts.append(
            f"\n⏳ Only {remaining} step(s) left — finish reviewing "
            "or submit with request_changes/approve."
        )
    else:
        parts.append("\nFind the next issue or submit your review. Respond with a single JSON object.")

    return "\n".join(parts)


def parse_action(response_text: str) -> dict:
    """Extract a JSON action from the model response."""
    text = response_text.strip()

    # Try direct JSON parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action_type" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # Try to find JSON within code blocks or raw braces
    json_pattern = re.compile(r"\{[^{}]*\"action_type\"[^{}]*\}", re.DOTALL)
    match = json_pattern.search(text)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj
        except json.JSONDecodeError:
            pass

    # Fallback: submit review with the raw text
    return {
        "action_type": "request_changes",
        "message": text[:500] if text else "Unable to parse response",
    }


# ── Main inference loop ─────────────────────────────────────────────

def run_task(client: OpenAI, env: EnvClient, task_id: str) -> None:
    """
    Run a single task episode with a proper multi-turn conversation.

    The LLM receives the full conversation history on every call, enabling
    genuine iterative reasoning rather than single-shot prompting.
    """
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    # Full conversation history passed to LLM every step
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        # Reset environment — get initial observation
        observation = env.reset(task_id)

        # First user message: the full PR context
        initial_prompt = build_initial_prompt(observation)
        messages.append({"role": "user", "content": initial_prompt})

        last_reward = 0.0

        for step_num in range(1, MAX_STEPS + 1):
            if observation.get("done", False):
                break

            # Call the model with full conversation history
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                print(f"[DEBUG] Model request failed: {exc}", flush=True)
                response_text = '{"action_type": "request_changes", "message": "Model error, submitting review"}'

            # Add assistant turn to conversation history
            messages.append({"role": "assistant", "content": response_text})

            # Parse action and send to environment
            action = parse_action(response_text)
            result = env.step(action)

            reward = _clamp(result.get("reward", 0.01))  # Default to valid min, never 0.0
            done = result.get("done", False)
            obs = result.get("observation", {})
            error = obs.get("last_action_error")

            rewards.append(reward)
            steps_taken = step_num
            last_reward = reward

            # Format action string for mandatory logging
            action_str = (
                f"{action.get('action_type', '?')}"
                f"({action.get('file_path', '')}:{action.get('line_number', '')})"
            )
            log_step(step=step_num, action=action_str, reward=reward, done=done, error=error)

            observation = obs

            if done:
                # Priority 4 FIX: success is based on the FINAL grader score
                # (the last reward when done=True is the full episode score 0.0–1.0)
                final_score = reward
                success = final_score >= SUCCESS_THRESHOLD
                break
            else:
                # Add follow-up user message with progress context
                followup = build_followup_prompt(obs, last_reward)
                messages.append({"role": "user", "content": followup})
        else:
            # Ran out of steps without done=True
            success = False

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} failed: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


def main() -> None:
    """Run inference on all tasks."""
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EnvClient(ENV_URL)

    try:
        tasks = env.get_tasks()
        for task in tasks:
            task_id = task["task_id"]
            print(f"\n{'='*60}", flush=True)
            print(f"Running task: {task_id} ({task['difficulty']})", flush=True)
            print(f"{'='*60}", flush=True)
            run_task(llm_client, env, task_id)
    finally:
        env.close()


if __name__ == "__main__":
    main()
