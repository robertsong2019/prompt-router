#!/usr/bin/env python3
"""
🧭 Prompt Router — Lightweight multi-agent prompt dispatcher.
Routes natural language tasks to the best-matching agent persona
using keyword/signature scoring. Zero dependencies.

Usage:
    python prompt_router.py "Fix the login bug"
    python prompt_router.py --list
    python prompt_router.py --add-agent
"""

import json
import re
import sys
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Agent:
    """An agent persona with routing signatures."""
    name: str
    description: str
    keywords: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)  # regex patterns
    priority: float = 1.0  # base multiplier

    def score(self, prompt: str, detail: bool = False) -> float | tuple[float, list[str]]:
        """Score how well this agent matches the prompt.
        If detail=True, returns (score, list_of_match_reasons).
        """
        prompt_lower = prompt.lower()
        prompt_words = set(re.findall(r'\w+', prompt_lower))
        score = 0.0
        reasons = []

        # Keyword matching (exact word overlap)
        for kw in self.keywords:
            kw_lower = kw.lower()
            kw_words = set(re.findall(r'\w+', kw_lower))
            overlap = prompt_words & kw_words
            if overlap:
                # TF-IDF-inspired: rarer matches score higher
                ratio = len(overlap) / len(kw_words)
                score += ratio * (1.0 + 0.1 * len(kw_words))
                reasons.append(f"keyword '{kw}' matched ({', '.join(overlap)})")

        # Regex pattern matching
        for pattern in self.patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            if matches:
                score += 0.5 * len(matches)
                reasons.append(f"pattern /{pattern}/ matched {len(matches)}x")

        final_score = score * self.priority
        if detail:
            return final_score, reasons
        return final_score


# --- Built-in agent registry ---

DEFAULT_AGENTS: list[Agent] = [
    Agent(
        name="coder",
        description="Writes, debugs, refactors, and reviews code",
        keywords=[
            "fix bug", "implement", "refactor", "code", "function", "class",
            "debug", "error", "compile", "test", "deploy", "api endpoint",
            "script", "algorithm", "optimize performance", "login", "auth",
            "database", "sql", "http", "request", "server",
        ],
        patterns=[
            r'\b\w+\.\w+\b',       # file.ext patterns
            r'\b(fix|bug|issue)\b', # issue keywords
            r'\b(write|create|build)\b.*\b(code|function|class|script)\b',
            r'`.+`',                # inline code
        ],
        priority=1.0,
    ),
    Agent(
        name="researcher",
        description="Searches, summarizes, and analyzes information",
        keywords=[
            "research", "find", "search", "summarize", "analyze", "compare",
            "explain", "what is", "how does", "why", "report", "survey",
            "data", "statistics", "trend", "market", "revenue",
        ],
        patterns=[
            r'\b(what|how|why|when|where)\b.*\?',
            r'\b(summarize|explain|analyze|compare)\b',
        ],
        priority=1.0,
    ),
    Agent(
        name="writer",
        description="Creates content, documentation, and copy",
        keywords=[
            "write", "draft", "document", "blog", "article", "email",
            "letter", "documentation", "readme", "guide", "tutorial",
            "story", "creative", "content", "copy", "announcement",
        ],
        patterns=[
            r'\b(write|draft|compose|create)\b.*\b(article|post|email|doc|letter)\b',
        ],
        priority=1.0,
    ),
    Agent(
        name="reviewer",
        description="Reviews, critiques, and improves existing work",
        keywords=[
            "review", "check", "verify", "validate", "audit", "improve",
            "feedback", "critique", "assess", "evaluate", "quality",
            "security", "vulnerability", "best practice",
        ],
        patterns=[
            r'\b(review|check|audit|verify)\b',
        ],
        priority=0.9,
    ),
    Agent(
        name="planner",
        description="Plans projects, breaks down tasks, estimates effort",
        keywords=[
            "plan", "schedule", "roadmap", "milestone", "task", "break down",
            "estimate", "timeline", "sprint", "backlog", "priority",
            "architecture", "design", "scope",
        ],
        patterns=[
            r'\b(plan|design|architect)\b.*\b(system|project|feature|app)\b',
        ],
        priority=0.95,
    ),
    Agent(
        name="translator",
        description="Translates text between languages",
        keywords=[
            "translate", "translation", "中文", "英文", "English", "Chinese",
            "日本語", "Japanese", "language", "本地化", "localize",
        ],
        patterns=[
            r'\btranslate\b.*\b(to|into|from)\b',
            r'[\u4e00-\u9fff]',  # Chinese characters in prompt
        ],
        priority=1.1,
    ),
]


class PromptRouter:
    """Routes prompts to the best-matching agent."""

    def __init__(self, agents: Optional[list[Agent]] = None):
        self.agents = agents or DEFAULT_AGENTS

    def route(self, prompt: str) -> tuple[str, float, list[tuple[str, float]]]:
        """
        Route a prompt to the best agent.
        Returns: (best_agent_name, best_score, all_scores)
        """
        scores = [(a.name, a.score(prompt)) for a in self.agents]
        scores.sort(key=lambda x: x[1], reverse=True)

        if not scores or scores[0][1] == 0:
            # Fallback: use simple heuristic on first word
            fallback = self._heuristic_fallback(prompt)
            return fallback, 0.0, scores

        # Normalize scores with softmax for nice display
        total = sum(math.exp(s) for _, s in scores[:5])
        normalized = [(name, math.exp(s) / total) for name, s in scores[:5]]

        return scores[0][0], scores[0][1], normalized

    def explain(self, prompt: str) -> dict:
        """Explain routing decision: which agent wins and why.
        Returns dict with agent, score, reasons per agent.
        """
        details = {}
        for a in self.agents:
            s, reasons = a.score(prompt, detail=True)
            details[a.name] = {"score": s, "reasons": reasons}

        best = max(details.items(), key=lambda x: x[1]["score"])
        return {
            "prompt": prompt,
            "best_agent": best[0] if best[1]["score"] > 0 else self._heuristic_fallback(prompt),
            "best_score": best[1]["score"],
            "agents": details,
        }

    def route_with_confidence(self, prompt: str, threshold: float = 0.5) -> tuple[Optional[str], float, str]:
        """Route with confidence check. Returns (agent, score, status).
        status is 'confident', 'low_confidence', or 'no_match'.
        Returns None agent when below threshold.
        """
        agent, score, _ = self.route(prompt)
        if score == 0.0:
            return None, 0.0, "no_match"
        if score < threshold:
            return agent, score, "low_confidence"
        return agent, score, "confident"

    def route_batch(self, prompts: list[str]) -> dict:
        """Route multiple prompts and return results + aggregate stats.
        Returns dict with 'results' list and 'stats' summary.
        """
        results = []
        for p in prompts:
            agent, score, all_scores = self.route(p)
            _, conf_score, status = self.route_with_confidence(p)
            results.append({
                "prompt": p,
                "agent": agent,
                "score": round(score, 4),
                "status": status,
            })

        # Aggregate stats
        distribution = {}
        status_counts = {}
        scores = []
        for r in results:
            distribution[r["agent"]] = distribution.get(r["agent"], 0) + 1
            status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
            scores.append(r["score"])

        stats = {
            "total": len(prompts),
            "distribution": distribution,
            "status_counts": status_counts,
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0,
            "max_score": round(max(scores), 4) if scores else 0,
            "min_score": round(min(scores), 4) if scores else 0,
        }

        return {"results": results, "stats": stats}

    def route_with_fallback(self, prompt: str, threshold: float = 0.5) -> dict:
        """Route with fallback chain: try agents in score order until one exceeds threshold.
        Returns dict with 'agent', 'score', 'fallback_used', 'chain'.
        """
        scores = [(a.name, a.score(prompt)) for a in self.agents]
        scores.sort(key=lambda x: x[1], reverse=True)

        chain = []
        for name, score in scores:
            chain.append({"agent": name, "score": round(score, 4)})
            if score >= threshold:
                return {
                    "agent": name,
                    "score": round(score, 4),
                    "fallback_used": len(chain) > 1,
                    "chain": chain,
                }

        # Nothing met threshold — return best available
        if scores:
            return {
                "agent": scores[0][0],
                "score": round(scores[0][1], 4),
                "fallback_used": True,
                "chain": chain,
            }

        return {"agent": None, "score": 0.0, "fallback_used": False, "chain": []}

    def add_agent(self, agent: Agent) -> None:
        """Add a custom agent to the router. No-op if name already exists."""
        if any(a.name == agent.name for a in self.agents):
            return
        self.agents.append(agent)

    def remove_agent(self, name: str) -> bool:
        """Remove an agent by name. Returns True if found and removed."""
        for i, a in enumerate(self.agents):
            if a.name == name:
                self.agents.pop(i)
                return True
        return False

    def list_agents(self) -> list[dict]:
        """Return summary of all registered agents."""
        return [{"name": a.name, "description": a.description,
                 "keywords": len(a.keywords), "priority": a.priority}
                for a in self.agents]

    def save_config(self, path: str) -> None:
        """Save current agent config to JSON file."""
        data = [{"name": a.name, "description": a.description,
                 "keywords": a.keywords, "patterns": a.patterns,
                 "priority": a.priority} for a in self.agents]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_config(self, path: str) -> None:
        """Load agent config from JSON file, replacing current agents."""
        with open(path) as f:
            data = json.load(f)
        self.agents = [Agent(name=d["name"], description=d["description"],
                             keywords=d.get("keywords", []),
                             patterns=d.get("patterns", []),
                             priority=d.get("priority", 1.0)) for d in data]

    def _heuristic_fallback(self, prompt: str) -> str:
        """Last-resort routing based on simple heuristics."""
        first = prompt.strip().split()[0].lower() if prompt.strip() else ""
        if first in ("write", "create", "build", "make"):
            return "coder"
        if first in ("what", "how", "why", "when"):
            return "researcher"
        return "coder"  # safe default


def main():
    if "--list" in sys.argv:
        print("Available agents:\n")
        for a in DEFAULT_AGENTS:
            print(f"  🤖 {a.name} — {a.description}")
            print(f"     keywords: {', '.join(a.keywords[:6])}...")
            print()
        return

    if not sys.argv[-1].startswith("-") and len(sys.argv) > 1:
        prompt = sys.argv[-1]
    else:
        prompt = input("Enter prompt: ").strip()

    router = PromptRouter()
    agent, score, all_scores = router.route(prompt)

    # Display results
    print(f"\n{'='*50}")
    print(f"  📨 Prompt: {prompt}")
    print(f"  🎯 Route → {agent} (score: {score:.2f})")
    print(f"{'='*50}")
    print(f"\n  Scoreboard:")
    for name, s in all_scores:
        bar = "█" * int(s * 30)
        print(f"    {name:12s} {bar} {s:.1%}")
    print()

    # Output JSON for programmatic use
    if "--json" in sys.argv:
        result = {
            "prompt": prompt,
            "agent": agent,
            "score": score,
            "all_scores": [(n, round(s, 4)) for n, s in all_scores],
        }
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
