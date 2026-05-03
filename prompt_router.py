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
        self.agents = list(agents) if agents is not None else list(DEFAULT_AGENTS)

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

    def route_top_k(self, prompt: str, k: int = 3) -> list[dict]:
        """Return top-K agent matches with scores and explanations.
        Useful for multi-agent delegation or ensemble routing.
        """
        scores = []
        for a in self.agents:
            s, reasons = a.score(prompt, detail=True)
            scores.append({"agent": a.name, "score": round(s, 4), "reasons": reasons})
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:k]

    def route_ensemble(self, prompt: str, k: int = 3, weights: Optional[dict[str, float]] = None) -> dict:
        """Route to multiple agents with weight distribution.
        Useful for splitting work across agents or building ensemble systems.
        Returns dict with 'agents' list (name, weight, score, reasons) and 'total_weight'.
        """
        scored = []
        for a in self.agents:
            s, reasons = a.score(prompt, detail=True)
            scored.append((a.name, s, reasons))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]

        # Apply external weight bias if provided
        total_raw = sum(s for _, s, _ in top) or 1.0
        agents_out = []
        for name, score, reasons in top:
            base_w = score / total_raw
            if weights and name in weights:
                base_w *= weights[name]
            agents_out.append({
                "agent": name,
                "score": round(score, 4),
                "weight": round(base_w, 4),
                "reasons": reasons,
            })

        # Re-normalize weights
        total_w = sum(a["weight"] for a in agents_out) or 1.0
        for a in agents_out:
            a["weight"] = round(a["weight"] / total_w, 4)

        return {"agents": agents_out, "total_weight": round(total_w, 4)}

    @staticmethod
    def merge_routers(*routers: 'PromptRouter') -> 'PromptRouter':
        """Merge multiple routers into one. Agents with same name are deduplicated (first wins).
        """
        if not routers:
            return PromptRouter([])
        seen = set()
        merged = []
        for r in routers:
            for a in r.agents:
                if a.name not in seen:
                    seen.add(a.name)
                    merged.append(a)
        return PromptRouter(merged)

    def route_adaptive(self, prompt: str, correct_agent: Optional[str] = None) -> dict:
        """Route with optional feedback. If correct_agent provided, boosts that agent's priority
        and records feedback. Returns routing result + feedback status.
        """
        if not hasattr(self, '_feedback_history'):
            self._feedback_history = []  # [(prompt_hash, agent, was_correct)]

        agent, score, all_scores = self.route(prompt)
        result = {"agent": agent, "score": round(score, 4), "feedback": None}

        if correct_agent:
            was_correct = agent == correct_agent
            self._feedback_history.append((hash(prompt.lower()), agent, was_correct))
            result["feedback"] = "correct" if was_correct else "corrected"

            if not was_correct:
                # Boost correct agent's priority
                for a in self.agents:
                    if a.name == correct_agent:
                        a.priority = min(a.priority + 0.1, 3.0)
                        break
                # Slightly reduce wrongly-selected agent
                for a in self.agents:
                    if a.name == agent:
                        a.priority = max(a.priority - 0.05, 0.1)
                        break

        result["history_size"] = len(self._feedback_history)
        result["accuracy"] = (
            sum(1 for _, _, ok in self._feedback_history if ok) / len(self._feedback_history)
            if self._feedback_history else None
        )
        return result

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

    def route_chain(self, prompts: list[str], diversify: bool = True,
                    penalty: float = 0.5) -> list[dict]:
        """Route a sequence of prompts, optionally diversifying across agents.
        Returns list of routing results with chain context (position, prev_agent).
        """
        history = []
        results = []
        for i, p in enumerate(prompts):
            if diversify and history:
                res = self.route_with_history(p, history, penalty=penalty)
                results.append({
                    "position": i,
                    "prompt": p,
                    "agent": res["agent"],
                    "score": res["score"],
                    "prev_agent": history[-1] if history else None,
                    "diversified": res["diversified"],
                })
            else:
                agent, score, _ = self.route(p)
                results.append({
                    "position": i,
                    "prompt": p,
                    "agent": agent,
                    "score": round(score, 4),
                    "prev_agent": history[-1] if history else None,
                    "diversified": False,
                })
            history.append(results[-1]["agent"])
        return results

    def score_matrix(self, prompts: list[str]) -> dict:
        """Compute a prompts × agents score matrix.
        Returns dict with 'matrix' (list of rows), 'agents' (column names),
        'best_agent_per_prompt', and 'agent_coverage' (% of prompts where agent wins).
        """
        agents = [a.name for a in self.agents]
        matrix = []
        best_per = []
        for p in prompts:
            row = []
            for a in self.agents:
                row.append(round(a.score(p), 4))
            matrix.append(row)
            best_idx = row.index(max(row))
            best_per.append(agents[best_idx] if max(row) > 0 else None)

        # Agent coverage: what % of prompts each agent wins
        coverage = {}
        for name in agents:
            wins = sum(1 for b in best_per if b == name)
            coverage[name] = round(wins / len(prompts), 4) if prompts else 0.0

        return {
            "agents": agents,
            "matrix": matrix,
            "best_agent_per_prompt": best_per,
            "agent_coverage": coverage,
        }

    def route_with_history(self, prompt: str, history: Optional[list[str]] = None,
                            avoid_repeat: bool = True, penalty: float = 0.3) -> dict:
        """Route considering conversation history. Avoids routing to agents
        that dominated recent turns, promoting diversity.
        Returns dict with 'agent', 'score', 'agent_counts', 'diversified'.
        """
        if history is None:
            history = []
        # Count recent agent usage
        counts: dict[str, int] = {}
        for h in history:
            counts[h] = counts.get(h, 0) + 1

        # Score all agents
        scores = [(a.name, a.score(prompt)) for a in self.agents]

        # Apply penalty for repeated agents
        adjusted = []
        for name, score in scores:
            adj = score
            if avoid_repeat and name in counts:
                adj -= penalty * counts[name]
            adjusted.append((name, adj, score))
        adjusted.sort(key=lambda x: x[1], reverse=True)

        best_name, best_adj, best_raw = adjusted[0] if adjusted else (None, 0.0, 0.0)
        if best_adj <= 0:
            # Pick best non-zero adjusted score
            positive = [(n, a, r) for n, a, r in adjusted if a > 0]
            if positive:
                best_name, best_adj, best_raw = positive[0]
            elif scores:
                scores.sort(key=lambda x: x[1], reverse=True)
                best_name, best_raw = scores[0]
                best_adj = best_raw

        return {
            "agent": best_name,
            "score": round(best_raw, 4),
            "adjusted_score": round(best_adj, 4),
            "agent_counts": counts,
            "diversified": best_name != self.route(prompt)[0] if counts else False,
        }

    def route_by_tags(self, prompt: str, tags: list[str]) -> tuple[Optional[str], float, list[tuple[str, float]]]:
        """Route only to agents whose name or description matches any of the given tags.
        Useful for domain-scoped routing (e.g. only route to code-related agents).
        Returns (agent, score, filtered_scores). Agent is None if no tag matches.
        """
        tags_lower = {t.lower() for t in tags}
        filtered = [a for a in self.agents
                    if a.name.lower() in tags_lower
                    or any(t in a.description.lower() for t in tags_lower)]
        if not filtered:
            return None, 0.0, []
        scores = [(a.name, a.score(prompt)) for a in filtered]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0] if scores[0][1] > 0 else None, scores[0][1], scores

    def route_round_robin(self, prompt: str, state: Optional[dict] = None) -> dict:
        """Route using weighted round-robin load balancing.
        State dict tracks per-agent assignment counts. Pass state between calls
        for continuity. Agent scores are used as weights — higher score = more likely
        to be picked, but less-loaded agents get a boost.
        Returns dict with 'agent', 'score', 'index', 'assignments'.
        """
        if state is None:
            state = {"index": 0, "assignments": {}}
        state.setdefault("assignments", {})
        state.setdefault("index", 0)

        # Score all agents
        scored = [(a.name, a.score(prompt)) for a in self.agents]
        if not scored:
            return {"agent": None, "score": 0.0, "index": 0, "assignments": {}}

        assignments = state["assignments"]
        max_assigned = max(assignments.values()) if assignments else 0

        # Boost underutilized agents: score - (count - min_count) * penalty
        counts = [assignments.get(n, 0) for n, _ in scored]
        min_count = min(counts) if counts else 0

        adjusted = []
        for i, (name, score) in enumerate(scored):
            load_penalty = (assignments.get(name, 0) - min_count) * 0.15
            adjusted.append((name, score - load_penalty, i))

        adjusted.sort(key=lambda x: x[1], reverse=True)
        best_name, best_score, best_idx = adjusted[0]

        # Update state
        assignments[best_name] = assignments.get(best_name, 0) + 1
        state["index"] = best_idx

        return {
            "agent": best_name,
            "score": round(best_score, 4),
            "index": best_idx,
            "assignments": dict(assignments),
        }

    def route_least_loaded(self, prompt: str, loads: Optional[dict[str, int]] = None,
                             threshold: float = 0.1) -> dict:
        """Route to the least-loaded agent that still meets a minimum score threshold.
        Loads dict tracks current load per agent. Among agents scoring above threshold,
        picks the one with lowest load. Ties broken by score.
        Returns dict with 'agent', 'score', 'load', 'candidates'.
        """
        if loads is None:
            loads = {}

        scored = [(a.name, a.score(prompt)) for a in self.agents]
        if not scored:
            return {"agent": None, "score": 0.0, "load": 0, "candidates": []}

        # Filter by threshold
        candidates = [(n, s) for n, s in scored if s >= threshold]
        if not candidates:
            # Fall back to best score regardless of threshold
            candidates = scored

        # Sort: least loaded first, then highest score
        candidates.sort(key=lambda x: (loads.get(x[0], 0), -x[1]))

        best_name, best_score = candidates[0]
        return {
            "agent": best_name,
            "score": round(best_score, 4),
            "load": loads.get(best_name, 0),
            "candidates": [(n, round(s, 4)) for n, s in candidates],
        }

    def route_by_capability(self, prompt: str, capabilities: list[str],
                              mode: str = "any") -> dict:
        """Route to agents whose keywords match required capabilities.
        mode='any': agent must match at least one capability (union)
        mode='all': agent must match all capabilities (intersection)
        mode='best': pick agent with most capability matches
        Returns dict with 'agent', 'score', 'matched_capabilities', 'mode'.
        """
        caps_lower = {c.lower() for c in capabilities}

        results = []
        for a in self.agents:
            kw_lower = {k.lower() for k in a.keywords}
            matched = caps_lower & kw_lower
            score = a.score(prompt)

            if mode == "any" and matched:
                results.append((a.name, score, matched))
            elif mode == "all" and matched == caps_lower:
                results.append((a.name, score, matched))
            elif mode == "best":
                results.append((a.name, score, matched))

        if not results:
            return {
                "agent": None, "score": 0.0,
                "matched_capabilities": set(), "mode": mode,
            }

        # Sort by number of matched capabilities (desc), then score (desc)
        results.sort(key=lambda x: (len(x[2]), x[1]), reverse=True)
        best_name, best_score, best_matched = results[0]

        return {
            "agent": best_name,
            "score": round(best_score, 4),
            "matched_capabilities": sorted(best_matched),
            "mode": mode,
        }

    def agent_stats(self) -> dict:
        """Return summary statistics about the router's agents.
        Includes per-agent keyword count, description length, and total agents.
        """
        agents_info = []
        for a in self.agents:
            agents_info.append({
                "name": a.name,
                "keyword_count": len(a.keywords),
                "description_length": len(a.description),
                "keywords": sorted(a.keywords),
            })
        return {
            "total_agents": len(self.agents),
            "agents": agents_info,
        }

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
