# 🧭 Prompt Router

A lightweight multi-agent prompt router in a single Python file. Takes a natural language task, scores it against registered agent personas, and routes to the best match — **without calling an LLM**.

## Why?

When you have multiple AI agent specialists (coder, researcher, writer, reviewer), you need a fast, **zero-cost** way to dispatch tasks to the right one. Using an LLM just for routing is wasteful. This does it with regex + keyword scoring in under 200 lines.

## Features

- **Zero dependencies** — stdlib only, Python 3.10+
- **Regex + keyword scoring** — TF-IDF-inspired matching
- **6 built-in agents** — coder, researcher, writer, reviewer, planner, translator
- **Softmax normalization** — clean probability display
- **CLI + library interface** — use interactively or programmatically
- **JSON output** — pipe into other tools with `--json`

## Quick Start

```bash
# Basic routing
python prompt_router.py "Fix the authentication bug in login.py"
# → routes to: coder (score: 2.10)

python prompt_router.py "Summarize the Q4 revenue report"
# → routes to: researcher (score: 1.30)

python prompt_router.py "Write a blog post about AI agents"
# → routes to: writer (score: 1.50)

# List available agents
python prompt_router.py --list

# JSON output for scripting
python prompt_router.py --json "Plan the migration to microservices"
```

## Library Usage

```python
from prompt_router import PromptRouter, Agent, DEFAULT_AGENTS

# Use built-in agents
router = PromptRouter()
agent, score, all_scores = router.route("Debug the memory leak in worker.py")
print(f"→ {agent} (score: {score:.2f})")

# Add a custom agent
custom = Agent(
    name="devops",
    description="Handles deployment, infrastructure, and CI/CD",
    keywords=["deploy", "docker", "kubernetes", "ci/cd", "pipeline", "infra"],
    patterns=[r'\b(docker|k8s|terraform|ansible)\b'],
    priority=1.2,
)
router = PromptRouter(agents=DEFAULT_AGENTS + [custom])
agent, score, _ = router.route("Set up the CI pipeline for staging")
print(f"→ {agent}")  # → devops
```

## How Scoring Works

1. **Keyword matching** — each agent has a keyword list; overlapping words with the prompt contribute to the score, weighted by keyword length (longer phrases = higher signal)
2. **Regex patterns** — agents define regex signatures (e.g., file extensions, code patterns) that add to the score on match
3. **Priority multiplier** — each agent has a base priority that scales the final score
4. **Softmax normalization** — top-5 scores are normalized for probability display
5. **Heuristic fallback** — if all scores are zero, falls back to first-word heuristic

## Output Example

```
==================================================
  📨 Prompt: Fix the authentication bug in login.py
  🎯 Route → coder (score: 2.10)
==================================================

  Scoreboard:
    coder        ████████████████████ 66.3%
    researcher   ██████               19.8%
    writer       ██                    5.2%
    reviewer     ██                    5.1%
    planner      █                     3.6%
```

## Built-in Agents

| Agent | Description | Priority |
|-------|-------------|----------|
| **coder** | Writes, debugs, refactors code | 1.0 |
| **researcher** | Searches, summarizes, analyzes info | 1.0 |
| **writer** | Creates docs, blog posts, emails | 1.0 |
| **reviewer** | Reviews, audits, validates work | 0.9 |
| **planner** | Plans projects, estimates timelines | 0.95 |
| **translator** | Translates between languages | 1.1 |

## File Structure

```
prompt-router/
├── prompt_router.py   # Everything in one file (~200 lines)
├── test_router.py     # Unit tests
└── README.md
```

## Extending

Add agents at runtime via `Agent` dataclass, or modify `DEFAULT_AGENTS` for permanent additions. Each agent needs:
- `name`, `description` — identity
- `keywords` — word/phrase list for matching
- `patterns` — regex list for structural matching
- `priority` — score multiplier (higher = more likely)

## License

MIT
