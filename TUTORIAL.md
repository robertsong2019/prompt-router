# Tutorial: Building Your Own Agent Router

> 30 minutes | Intermediate

You'll learn how to use and extend `prompt-router` to build a custom multi-agent dispatch system.

## What You'll Learn

- How the scoring algorithm works
- How to add custom agents
- How to integrate routing into a larger system

## Prerequisites

- Python 3.10+
- `prompt_router.py` from this directory

## Part 1: Understanding the Scoring (5 min)

Each agent scores a prompt in two ways:

**Keyword matching** — The agent's keyword list is compared against words in the prompt. Longer keyword phrases score higher (TF-IDF-inspired).

```python
# "Fix the authentication bug" matches coder keyword "fix bug"
# Score += overlap_ratio * (1 + 0.1 * keyword_length)
```

**Regex patterns** — Structural patterns like file extensions (`login.py`) or code ticks (`` `code` ``) add bonus points.

```python
# Prompt "Fix bug in login.py" matches pattern r'\b\w+\.\w+\b'
# Score += 0.5 per match
```

Try it yourself:

```bash
python prompt_router.py "Fix bug in login.py"
python prompt_router.py "What is machine learning?"
python prompt_router.py "Write a tutorial about Docker"
```

Notice how each routes to a different agent based on content, not just keywords.

## Part 2: Adding a Custom Agent (10 min)

Let's add a "data scientist" agent:

```python
from prompt_router import PromptRouter, Agent, DEFAULT_AGENTS

data_scientist = Agent(
    name="data-scientist",
    description="Handles data analysis, ML models, and statistics",
    keywords=[
        "train model", "dataset", "accuracy", "precision", "recall",
        "regression", "classification", "neural network", "feature",
        "pandas", "numpy", "jupyter", "notebook", "visualization",
        "correlation", "distribution", "outlier", "preprocessing",
    ],
    patterns=[
        r'\b(train|fit|predict|evaluate)\b.*\b(model|classifier|regressor)\b',
        r'\b\.(csv|parquet|jsonl)\b',   # data file extensions
        r'\b(pandas|numpy|sklearn|torch)\b',
    ],
    priority=1.0,
)

router = PromptRouter(agents=DEFAULT_AGENTS + [data_scientist])
agent, score, _ = router.route("Train a random forest on the housing dataset")
print(f"→ {agent}")  # → data-scientist
```

**Tips for good keywords:**
- Use multi-word phrases ("train model" not just "train")
- Include domain-specific tools/libraries
- Cover synonyms ("dataset", "data", "corpus")

**Tips for good patterns:**
- Match file extensions relevant to the domain
- Match tool/library names
- Match domain-specific verb phrases

## Part 3: Building a Dispatch Pipeline (10 min)

Now let's wire the router into a simple dispatch loop:

```python
from prompt_router import PromptRouter, DEFAULT_AGENTS

# Simulated agent handlers
handlers = {
    "coder": lambda task: f"[CODE] Implementing: {task}",
    "researcher": lambda task: f"[RESEARCH] Looking into: {task}",
    "writer": lambda task: f"[WRITE] Drafting: {task}",
}

router = PromptRouter()

def dispatch(task: str) -> str:
    agent, score, _ = router.route(task)
    
    # Reject low-confidence routes
    if score < 0.3:
        return f"⚠️ Low confidence (score={score:.2f}). Please clarify your request."
    
    handler = handlers.get(agent)
    if not handler:
        return f"⚠️ No handler for agent '{agent}'"
    
    return handler(task)

# Test
print(dispatch("Fix the memory leak in worker.py"))
print(dispatch("Research the latest RLHF techniques"))
print(dispatch("Write release notes for v2.0"))
```

## Part 4: Integration Patterns (5 min)

### With an actual LLM backend

```python
import openai

def call_agent(agent_name: str, prompt: str) -> str:
    system_prompts = {
        "coder": "You are an expert software engineer.",
        "researcher": "You are a thorough research analyst.",
        "writer": "You are a skilled technical writer.",
    }
    resp = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompts[agent_name]},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content
```

### As an MCP tool

Wrap `PromptRouter.route()` as an MCP tool so other agents can call it for delegation.

### In a web API

```python
from fastapi import FastAPI
from prompt_router import PromptRouter

app = FastAPI()
router = PromptRouter()

@app.post("/route")
def route_task(prompt: str):
    agent, score, scores = router.route(prompt)
    return {"agent": agent, "confidence": score, "alternatives": scores}
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Wrong agent selected | Add more specific keywords/patterns to the correct agent |
| Low scores for everything | Check that keywords are lowercase and use common phrasing |
| Tied scores | Adjust `priority` to break ties (higher = preferred) |
| Fallback triggers | The prompt doesn't match any keywords — add broader patterns |

## Summary

You now know how to:
- ✅ Route prompts to agents using keyword + regex scoring
- ✅ Add custom agents with tailored signatures
- ✅ Build dispatch pipelines with confidence thresholds
- ✅ Integrate routing into LLM backends and web APIs

## Next Steps

- Experiment with different `priority` values to fine-tune routing
- Add agents for your specific use cases (DevOps, security, legal, etc.)
- Combine with `agent-context-store` for persistent agent memory
