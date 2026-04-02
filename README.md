---
title: OpenEnv CodeReview
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# CodeReview OpenEnv 🔍

**An OpenEnv environment for AI-assisted code review evaluation.**

Agents review pull request diffs, identify bugs and security vulnerabilities, and submit structured reviews. Features 3 difficulty-graded tasks with deterministic graders — from spotting obvious null dereferences to catching subtle SQL injection patterns.

## 🎯 Why Code Review?

Code review is one of the most common and impactful tasks in software engineering. Every day, millions of developers review pull requests looking for bugs, security issues, and code quality problems. Training AI agents for this task has immediate, real-world value:

- **CI/CD bots** that automatically review PRs before human review
- **IDE copilots** that catch issues in real-time as developers write code
- **Security audit tools** that identify vulnerabilities at scale

## 📋 Tasks

| Task ID | Difficulty | Description | Key Challenge |
|---------|-----------|-------------|---------------|
| `find-obvious-bug` | ⭐ Easy | Find a null-dereference bug in a user profile utility | Single bug — calling `.strip()` on a potentially `None` value |
| `triage-mixed-pr` | ⭐⭐ Medium | Triage a multi-file PR with 3 issues of varying severity | Race condition (critical), boundary validation (major), unused import (minor) |
| `security-audit` | ⭐⭐⭐ Hard | Audit an admin API endpoint for security vulnerabilities | SQL injection, hardcoded secrets, missing authentication |

## 🔄 Environment API

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `string` | Active task identifier |
| `pr_title` | `string` | Pull request title |
| `pr_description` | `string` | PR description/body |
| `files` | `FileChange[]` | Changed files with diffs and full content |
| `existing_comments` | `ReviewComment[]` | Comments placed so far |
| `step` | `int` | Current step (1-indexed) |
| `max_steps` | `int` | Maximum steps (default: 10) |
| `last_action_error` | `string?` | Error from previous action |
| `done` | `bool` | Whether episode has ended |

### Action Space

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `"comment" \| "approve" \| "request_changes"` | Type of review action |
| `file_path` | `string?` | Target file (required for comments) |
| `line_number` | `int?` | Target line (required for comments) |
| `message` | `string` | Comment body or review summary |
| `severity` | `"critical" \| "major" \| "minor" \| "suggestion"` | Issue severity |

### Reward Signal

- **Incremental**: Small reward (+0.05 × match quality) per comment that matches a ground-truth issue
- **Final**: Full grader score (0.0–1.0) when the agent submits the review

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start a new episode with `{ "task_id": "..." }` |
| `POST` | `/step` | Take an action, returns `{ observation, reward, done, info }` |
| `GET` | `/state` | Full serializable environment state |
| `GET` | `/tasks` | List available tasks |
| `GET` | `/health` | Health check |

## 🚀 Setup

### Local Development

```bash
# Clone the repository
git clone https://huggingface.co/spaces/chandan123467896uyjh/openenv-codereview
cd openenv-codereview

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn codereview_env.server:app --host 0.0.0.0 --port 7860

# Test it
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "find-obvious-bug"}'
```

### Docker

```bash
# Build
docker build -t codereview-env .

# Run
docker run -p 7860:7860 codereview-env

# Verify
curl http://localhost:7860/health
```

### Run Baseline Inference

```bash
# Set environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="your-token"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export ENV_URL="http://localhost:7860"

# Run inference on all tasks
python inference.py
```

## 📊 Baseline Scores

Scores from the baseline inference script using Qwen/Qwen2.5-72B-Instruct:

| Task | Difficulty | Expected Score Range |
|------|-----------|---------------------|
| `find-obvious-bug` | Easy | 0.60 – 1.00 |
| `triage-mixed-pr` | Medium | 0.40 – 0.80 |
| `security-audit` | Hard | 0.30 – 0.70 |

*Actual scores may vary based on model temperature and API availability.*

## 🏗️ Architecture

```
openenv-codereview/
├── codereview_env/
│   ├── models.py    # Pydantic: Observation, Action, Reward
│   ├── env.py       # Core: reset(), step(), state()
│   ├── tasks.py     # 3 tasks + deterministic graders
│   ├── server.py    # FastAPI REST server
│   └── data/        # PR diff JSON files
├── inference.py     # Baseline agent script
├── openenv.yaml     # OpenEnv metadata
├── Dockerfile       # Docker containerization
└── tests/           # Unit tests
```

## 🧪 Grading

Each task has a **deterministic grader** that:
1. Compares agent comments against ground-truth bug annotations
2. Uses **keyword matching** (does the comment mention the right issue?)
3. Uses **line-number proximity** (within ±5 lines of the ground truth)
4. Returns a score between 0.0 and 1.0

### Scoring Rules

- **Full match** (correct file + line + keywords): 100% of issue weight
- **Partial match** (correct file + keywords, wrong line): 60%
- **Weak match** (keywords only): 30%
- **Severity bonus** (hard task only): +3.3% per correct severity tag

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

## 👤 Author

**chandan123467896uyjh** — Built for the OpenEnv Hackathon.
