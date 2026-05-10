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
        self._cooldowns: dict[str, float] = {}  # agent_name -> cooldown_until timestamp

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

    def route_with_diversity(self, prompt: str, recent_agents: Optional[list[str]] = None,
                              penalty: float = 0.3) -> dict:
        """Route with diversity penalty — discourages recently-used agents.
        Args:
            recent_agents: list of agent names used recently
            penalty: score reduction factor per recent occurrence
        Returns dict with agent, score, penalties, all_scored.
        """
        if recent_agents is None:
            recent_agents = []
        counts = {}
        for name in recent_agents:
            counts[name] = counts.get(name, 0) + 1

        all_scored = []
        for a in self.agents:
            base = a.score(prompt)
            pen = counts.get(a.name, 0) * penalty
            adjusted = max(0.0, base - pen)
            all_scored.append({
                "name": a.name,
                "base_score": base,
                "penalty": pen,
                "adjusted_score": adjusted,
            })

        all_scored.sort(key=lambda x: x["adjusted_score"], reverse=True)
        best = all_scored[0] if all_scored else None

        if not best or best["adjusted_score"] == 0.0:
            fallback = self._heuristic_fallback(prompt)
            # find fallback in scored list or create entry
            fb_entry = next((s for s in all_scored if s["name"] == fallback), None)
            if fb_entry:
                best = fb_entry
            else:
                best = {"name": fallback, "base_score": 0.0, "penalty": 0.0, "adjusted_score": 0.0}

        return {
            "agent": best["name"],
            "base_score": best["base_score"],
            "penalty": best["penalty"],
            "adjusted_score": best["adjusted_score"],
            "all_scored": all_scored,
        }

    def deduplicate_agents(self) -> int:
        """Remove duplicate agents (by name), keeping first occurrence.
        Returns number of duplicates removed.
        """
        seen = set()
        unique = []
        for a in self.agents:
            if a.name not in seen:
                seen.add(a.name)
                unique.append(a)
        removed = len(self.agents) - len(unique)
        self.agents = unique
        return removed

    def route_by_regex(self, prompt: str, pattern: str) -> dict:
        """Route to agent whose keywords match a regex pattern.
        Score = relevance score × (number of matching keywords / total keywords).
        Returns dict with agent, score, matches, all_matches.
        """
        import re
        regex = re.compile(pattern, re.IGNORECASE)
        all_matches = []
        for a in self.agents:
            matched_kws = [kw for kw in a.keywords if regex.search(kw)]
            if matched_kws:
                base = a.score(prompt)
                bonus = len(matched_kws) / max(len(a.keywords), 1)
                score = base * (0.5 + 0.5 * bonus)  # weight: 50% relevance + 50% match ratio
                all_matches.append({
                    "name": a.name,
                    "score": score,
                    "matched_keywords": matched_kws,
                    "match_ratio": bonus,
                })

        all_matches.sort(key=lambda x: x["score"], reverse=True)
        best = all_matches[0] if all_matches else None

        if not best or best["score"] == 0.0:
            best = {"name": None, "score": 0.0, "matched_keywords": [], "match_ratio": 0.0}

        return {
            "agent": best["name"],
            "score": best["score"],
            "matched_keywords": best["matched_keywords"],
            "match_ratio": best["match_ratio"],
            "all_matches": all_matches,
        }

    def cooldown(self, name: str, seconds: float = 60.0) -> bool:
        """Temporarily disable an agent for `seconds` seconds.
        Returns True if agent found and cooled down, False otherwise.
        """
        import time
        if not any(a.name == name for a in self.agents):
            return False
        self._cooldowns[name] = time.time() + seconds
        return True

    def is_cooled_down(self, name: str) -> bool:
        """Check if an agent is currently in cooldown. Returns True if cooled down (inactive)."""
        import time
        deadline = self._cooldowns.get(name)
        if deadline is None:
            return False
        if time.time() >= deadline:
            del self._cooldowns[name]
            return False
        return True

    def clear_cooldown(self, name: str) -> bool:
        """Remove cooldown for an agent. Returns True if cooldown was active."""
        return self._cooldowns.pop(name, None) is not None

    def route_respecting_cooldowns(self, prompt: str) -> tuple[Optional[str], float, list[tuple[str, float]]]:
        """Route to best agent that is not in cooldown.
        Returns (agent, score, all_eligible_scores). Agent is None if all cooled down.
        """
        eligible = [(a.name, a.score(prompt)) for a in self.agents
                    if not self.is_cooled_down(a.name)]
        eligible.sort(key=lambda x: x[1], reverse=True)

        if not eligible or eligible[0][1] == 0:
            # Use fallback only if the fallback agent is not cooled down
            fallback = self._heuristic_fallback(prompt)
            if fallback and not self.is_cooled_down(fallback):
                return fallback, 0.0, eligible
            return None, 0.0, eligible

        return eligible[0][0], eligible[0][1], eligible

    def route_by_priority(self, prompt: str, boost: float = 0.0, decay: float = 0.0,
                           history: Optional[list[str]] = None) -> dict:
        """Route weighted heavily by agent priority field.
        boost: add to priority for each match (keyword/pattern hit)
        decay: multiply priority by decay^(times agent used in history)
        Returns dict with 'agent', 'effective_priority', 'rankings'.
        """
        if history is None:
            history = []
        usage_counts: dict[str, int] = {}
        for h in history:
            usage_counts[h] = usage_counts.get(h, 0) + 1

        rankings = []
        for a in self.agents:
            base = a.priority
            score = a.score(prompt)
            effective = base + boost * (1.0 if score > 0 else 0.0)
            if decay > 0 and a.name in usage_counts:
                effective *= decay ** usage_counts[a.name]
            rankings.append({
                "name": a.name,
                "base_priority": base,
                "score": round(score, 4),
                "effective_priority": round(effective, 4),
                "usage": usage_counts.get(a.name, 0),
            })

        rankings.sort(key=lambda x: (x["effective_priority"], x["score"]), reverse=True)
        best = rankings[0] if rankings else None

        return {
            "agent": best["name"] if best else None,
            "effective_priority": best["effective_priority"] if best else 0.0,
            "rankings": rankings,
        }

    def route_tournament(self, prompt: str, rounds: int = 0) -> dict:
        """Bracket-style elimination routing.
        Agents compete in pairs; winner advances. Continues until one champion remains.
        rounds: if > 0, limits bracket rounds (e.g. 1 = just one comparison).
                if 0, runs full elimination.
        Returns dict with 'champion', 'final_score', 'bracket' (list of rounds),
        and 'rounds_played'.
        """
        import math as _math

        if not self.agents:
            return {"champion": None, "final_score": 0.0, "bracket": [], "rounds_played": 0}

        # Seed: sort agents by base priority for consistent bracket
        seeds = sorted(self.agents, key=lambda a: a.priority, reverse=True)
        scores = {a.name: a.score(prompt) for a in seeds}

        bracket = []
        current = list(seeds)
        rounds_played = 0
        max_rounds = rounds if rounds > 0 else 999

        while len(current) > 1 and rounds_played < max_rounds:
            next_round = []
            round_matches = []
            i = 0
            while i + 1 < len(current):
                a, b = current[i], current[i + 1]
                sa, sb = scores[a.name], scores[b.name]
                winner = a if sa >= sb else b
                loser = b if sa >= sb else a
                round_matches.append({
                    "match": f"{a.name} vs {b.name}",
                    "scores": {a.name: round(sa, 4), b.name: round(sb, 4)},
                    "winner": winner.name,
                    "loser": loser.name,
                })
                next_round.append(winner)
                i += 2
            # Odd agent gets a bye
            if i < len(current):
                bye = current[i]
                round_matches.append({
                    "match": f"{bye.name} (bye)",
                    "scores": {bye.name: round(scores[bye.name], 4)},
                    "winner": bye.name,
                    "loser": None,
                })
                next_round.append(bye)
            bracket.append(round_matches)
            current = next_round
            rounds_played += 1

        champion = current[0] if current else None
        return {
            "champion": champion.name if champion else None,
            "final_score": round(scores[champion.name], 4) if champion else 0.0,
            "bracket": bracket,
            "rounds_played": rounds_played,
        }

    def agent_similarity(self, name_a: str, name_b: str) -> dict:
        """Measure keyword overlap between two agents.
        Returns dict with 'jaccard' (intersection/union), 'overlap_count',
        'shared_keywords', and 'unique_to_a'/'unique_to_b'.
        Agent not found → empty dict.
        """
        a = next((x for x in self.agents if x.name == name_a), None)
        b = next((x for x in self.agents if x.name == name_b), None)
        if not a or not b:
            return {}

        set_a = {k.lower() for k in a.keywords}
        set_b = {k.lower() for k in b.keywords}
        shared = set_a & set_b
        union = set_a | set_b
        jaccard = len(shared) / len(union) if union else 0.0

        return {
            "jaccard": round(jaccard, 4),
            "overlap_count": len(shared),
            "shared_keywords": sorted(shared),
            "unique_to_a": sorted(set_a - set_b),
            "unique_to_b": sorted(set_b - set_a),
        }

    def route_weighted_vote(self, prompt: str,
                              strategies: Optional[list[str]] = None,
                              weights: Optional[dict[str, float]] = None) -> dict:
        """Multi-strategy voting ensemble for routing.
        Each strategy votes for its top agent. Votes are weighted and tallied.
        Built-in strategies: 'score', 'top_k', 'diversity', 'priority'.
        Returns dict with 'winner', 'tally', 'votes', 'strategies_used'.
        """
        if strategies is None:
            strategies = ["score", "top_k"]
        if weights is None:
            weights = {s: 1.0 for s in strategies}

        tally: dict[str, float] = {}
        votes = []

        for strat in strategies:
            w = weights.get(strat, 1.0)
            if strat == "score":
                agent, score, _ = self.route(prompt)
                tally[agent] = tally.get(agent, 0) + w
                votes.append({"strategy": "score", "agent": agent, "score": round(score, 4), "weight": w})
            elif strat == "top_k":
                top = self.route_top_k(prompt, k=1)
                if top:
                    a = top[0]["agent"]
                    tally[a] = tally.get(a, 0) + w
                    votes.append({"strategy": "top_k", "agent": a, "weight": w})
            elif strat == "diversity":
                div = self.route_with_diversity(prompt)
                a = div["agent"]
                tally[a] = tally.get(a, 0) + w
                votes.append({"strategy": "diversity", "agent": a, "weight": w})
            elif strat == "priority":
                pri = self.route_by_priority(prompt)
                a = pri["agent"]
                tally[a] = tally.get(a, 0) + w
                votes.append({"strategy": "priority", "agent": a, "weight": w})

        if not tally:
            # Fallback
            agent, _, _ = self.route(prompt)
            tally[agent] = 1.0

        winner = max(tally, key=tally.get)  # type: ignore[arg-type]
        return {
            "winner": winner,
            "tally": {k: round(v, 4) for k, v in sorted(tally.items(), key=lambda x: -x[1])},
            "votes": votes,
            "strategies_used": strategies,
        }

    def export_report(self, prompts: list[str], path: Optional[str] = None) -> dict:
        """Generate a routing analysis report for a set of prompts.
        Includes per-prompt routing, score matrix summary, agent coverage,
        confidence distribution, and diversity score.
        If path is given, writes JSON report to file.
        Returns the report dict.
        """
        report = {
            "total_prompts": len(prompts),
            "agents": [a.name for a in self.agents],
            "per_prompt": [],
            "agent_coverage": {},
            "confidence_distribution": {"confident": 0, "low_confidence": 0, "no_match": 0},
        }

        for p in prompts:
            agent, score, _ = self.route(p)
            _, conf_score, status = self.route_with_confidence(p)
            report["per_prompt"].append({
                "prompt": p,
                "agent": agent,
                "score": round(score, 4),
                "confidence": status,
            })
            report["agent_coverage"][agent] = report["agent_coverage"].get(agent, 0) + 1
            report["confidence_distribution"][status] += 1

        # Diversity: how evenly are agents used (entropy-based)
        import math as _math
        total = len(prompts) or 1
        counts = list(report["agent_coverage"].values())
        if counts:
            probs = [c / total for c in counts]
            entropy = -sum(p * _math.log2(p) for p in probs if p > 0)
            max_entropy = _math.log2(len(self.agents)) if len(self.agents) > 1 else 1.0
            report["diversity_score"] = round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0
        else:
            report["diversity_score"] = 0.0

        if path:
            with open(path, "w") as f:
                json.dump(report, f, indent=2)

        return report

    def health_check(self) -> dict:
        """Diagnostic health check for the router configuration.
        Returns issues list (empty = healthy) and summary stats.
        Checks: no agents, duplicate names, agents with no keywords,
        agents with zero-weight keywords, cooldowns active.
        """
        issues = []
        names = [a.name for a in self.agents]
        seen = set()
        for n in names:
            if n in seen:
                issues.append(f"duplicate agent name: {n}")
            seen.add(n)
        for a in self.agents:
            if not a.keywords:
                issues.append(f"agent '{a.name}' has no keywords")
            if not a.patterns and not a.keywords:
                issues.append(f"agent '{a.name}' has no routing signals (keywords or patterns)")
            if not a.description:
                issues.append(f"agent '{a.name}' has no description")
        active_cooldowns = [n for n in names if self.is_cooled_down(n)] if hasattr(self, '_cooldowns') else []
        for n in active_cooldowns:
            issues.append(f"agent '{n}' is in cooldown")
        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "agent_count": len(self.agents),
            "active_cooldowns": active_cooldowns,
        }

    def route_by_time_window(self, prompt: str, windows: dict[str, tuple[int, int]],
                              current_hour: Optional[int] = None) -> dict:
        """Route only to agents whose time window covers current hour.
        windows: {agent_name: (start_hour, end_hour)} in 0-23.
        start <= current_hour < end (half-open).
        Agents not in windows dict are always eligible.
        """
        import datetime as _dt
        if current_hour is None:
            current_hour = _dt.datetime.now().hour
        eligible = []
        for a in self.agents:
            if a.name not in windows:
                eligible.append(a)
            else:
                start, end = windows[a.name]
                if start <= end:
                    if start <= current_hour < end:
                        eligible.append(a)
                else:  # wraps midnight, e.g. (22, 6)
                    if current_hour >= start or current_hour < end:
                        eligible.append(a)
        if not eligible:
            return {"agent": None, "score": 0.0, "reason": "no agents available in this time window",
                    "current_hour": current_hour}
        temp = PromptRouter(eligible)
        agent, score, all_scores = temp.route(prompt)
        return {
            "agent": agent, "score": round(score, 4),
            "current_hour": current_hour,
            "eligible_count": len(eligible),
            "all_scores": [(n, round(s, 4)) for n, s in all_scores],
        }

    def cross_validate(self, test_cases: list[tuple[str, str]]) -> dict:
        """Evaluate routing accuracy against labeled test data.
        
        Args:
            test_cases: list of (prompt, expected_agent) tuples.
        
        Returns dict with accuracy, per_agent metrics, confusion matrix.
        """
        if not test_cases:
            return {"total": 0, "correct": 0, "accuracy": 0.0,
                    "per_agent": {}, "confusion_matrix": {}, "errors": []}

        correct = 0
        errors = []
        agent_names = [a.name for a in self.agents]
        # confusion_matrix[predicted][actual] = count
        confusion: dict[str, dict[str, int]] = {p: {a: 0 for a in agent_names} for p in agent_names}
        per_agent: dict[str, dict] = {a: {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
                                       for a in agent_names}

        for prompt, expected in test_cases:
            predicted, score, _ = self.route(prompt)
            if expected in confusion.get(predicted, {}):
                confusion[predicted][expected] += 1
            if predicted == expected:
                correct += 1
                if expected in per_agent:
                    per_agent[expected]["tp"] += 1
            else:
                errors.append({"prompt": prompt, "expected": expected, "predicted": predicted, "score": round(score, 4)})
                if predicted in per_agent:
                    per_agent[predicted]["fp"] += 1
                if expected in per_agent:
                    per_agent[expected]["fn"] += 1

        # compute precision/recall/f1
        for name, m in per_agent.items():
            m["precision"] = round(m["tp"] / (m["tp"] + m["fp"]), 4) if (m["tp"] + m["fp"]) > 0 else 0.0
            m["recall"] = round(m["tp"] / (m["tp"] + m["fn"]), 4) if (m["tp"] + m["fn"]) > 0 else 0.0
            m["f1"] = round(2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"]), 4) \
                if (m["precision"] + m["recall"]) > 0 else 0.0

        return {
            "total": len(test_cases),
            "correct": correct,
            "accuracy": round(correct / len(test_cases), 4),
            "per_agent": per_agent,
            "confusion_matrix": confusion,
            "errors": errors,
        }

    def suggest_improvements(self, test_cases: list[tuple[str, str]], top_k: int = 5) -> dict:
        """Analyze misclassified prompts and suggest keyword improvements.
        
        Args:
            test_cases: labeled (prompt, expected_agent) data.
            top_k: max suggestions per agent.
        
        Returns suggestions for underperforming agents.
        """
        cv = self.cross_validate(test_cases)
        if cv["accuracy"] == 1.0:
            return {"accuracy": 1.0, "suggestions": [], "message": "Perfect accuracy, no improvements needed."}

        suggestions = []
        agent_keywords = {a.name: set(kw.lower() for kw in a.keywords) for a in self.agents}

        for error in cv["errors"]:
            prompt_words = set(re.findall(r'\w+', error["prompt"].lower()))
            expected = error["expected"]
            # words not in expected agent's keywords that could help
            if expected in agent_keywords:
                missing = prompt_words - agent_keywords[expected]
                # filter short/generic words
                useful = [w for w in missing if len(w) >= 4]
                if useful:
                    suggestions.append({
                        "agent": expected,
                        "misclassified_as": error["predicted"],
                        "prompt": error["prompt"],
                        "suggested_keywords": useful[:top_k],
                    })

        # aggregate by agent
        by_agent: dict[str, list] = {}
        for s in suggestions:
            by_agent.setdefault(s["agent"], []).append(s)

        return {
            "accuracy": cv["accuracy"],
            "total_errors": len(cv["errors"]),
            "suggestions": suggestions,
            "by_agent": by_agent,
        }

    def route_negotiation(self, prompt: str, top_k: int = 3) -> dict:
        """Two-phase negotiation routing for ambiguous prompts.
        Phase 1: Score all agents, pick top-K candidates.
        Phase 2: Among candidates, re-score with tiebreaker heuristics
        (description overlap, specificity = fewer keywords = more specialized).
        Returns dict with 'winner', 'candidates', 'phase1_scores', 'phase2_scores',
        'negotiated' (True if winner changed from phase 1 to phase 2).
        """
        # Phase 1: standard scoring
        all_scored = []
        for a in self.agents:
            s, reasons = a.score(prompt, detail=True)
            all_scored.append((a.name, s, reasons))
        all_scored.sort(key=lambda x: x[1], reverse=True)

        candidates = all_scored[:top_k]
        phase1_winner = candidates[0][0] if candidates else None

        # Phase 2: re-score with tiebreakers
        prompt_words = set(re.findall(r'\w+', prompt.lower()))
        phase2 = []
        for name, base_score, reasons in candidates:
            agent = next(a for a in self.agents if a.name == name)
            # Description overlap bonus
            desc_words = set(re.findall(r'\w+', agent.description.lower()))
            desc_overlap = len(prompt_words & desc_words) * 0.3
            # Specificity bonus: fewer keywords = more specialized = higher bonus
            specificity = 1.0 / max(len(agent.keywords), 1)
            specificity_bonus = specificity * 0.2 if base_score > 0 else 0
            # Pattern match bonus
            pattern_hits = sum(len(re.findall(p, prompt, re.IGNORECASE)) for p in agent.patterns)
            pattern_bonus = pattern_hits * 0.15

            final = base_score + desc_overlap + specificity_bonus + pattern_bonus
            phase2.append({
                "agent": name,
                "base_score": round(base_score, 4),
                "desc_overlap": round(desc_overlap, 4),
                "specificity_bonus": round(specificity_bonus, 4),
                "pattern_bonus": round(pattern_bonus, 4),
                "final_score": round(final, 4),
            })

        phase2.sort(key=lambda x: x["final_score"], reverse=True)
        winner = phase2[0]["agent"] if phase2 else None

        return {
            "winner": winner,
            "candidates": [c["agent"] for c in phase2],
            "phase1_scores": [(n, round(s, 4)) for n, s, _ in all_scored],
            "phase2_scores": phase2,
            "negotiated": winner != phase1_winner,
        }

    def route_by_sentiment(self, prompt: str) -> dict:
        """Classify prompt sentiment/urgency and route accordingly.
        Detects: urgency (ASAP, urgent, critical), politeness (please, kindly),
        frustration (broken, broken again, still not working), question vs command.
        Returns dict with 'agent', 'score', 'sentiment', 'urgency', 'signals'.
        Sentiment-aware agents: urgent+code → coder(reviewer may be too slow),
        frustrated → reviewer (for audit), polite_question → researcher.
        """
        import re as _re
        p_lower = prompt.lower()

        signals = {
            "urgent": bool(_re.search(r'\b(asap|urgent|critical|emergency|immediately|right now)\b', p_lower)),
            "polite": bool(_re.search(r'\b(please|kindly|could you|would you)\b', p_lower)),
            "frustrated": bool(_re.search(r'\b(broken|still not|again|keeps failing|sick of|tired of)\b', p_lower)),
            "question": bool(_re.search(r'\?|\b(how|what|why|when|where)\b', p_lower)),
            "command": bool(_re.search(r'^(fix|write|create|build|deploy|implement|delete|remove|add)\b', p_lower)),
        }

        # Count active signals
        active = [k for k, v in signals.items() if v]

        # Determine sentiment label
        if signals["urgent"]:
            sentiment = "urgent"
        elif signals["frustrated"]:
            sentiment = "frustrated"
        elif signals["polite"] and signals["question"]:
            sentiment = "polite_question"
        elif signals["question"]:
            sentiment = "question"
        elif signals["command"]:
            sentiment = "command"
        else:
            sentiment = "neutral"

        # Sentiment-based agent preferences
        preference_boost: dict[str, float] = {}
        if sentiment == "urgent":
            preference_boost = {"coder": 0.5, "planner": 0.3}
        elif sentiment == "frustrated":
            preference_boost = {"reviewer": 0.4, "coder": 0.3}
        elif sentiment == "polite_question":
            preference_boost = {"researcher": 0.4, "writer": 0.2}
        elif sentiment == "question":
            preference_boost = {"researcher": 0.3}
        elif sentiment == "command":
            preference_boost = {"coder": 0.3, "writer": 0.2}

        # Score agents with sentiment boost
        scored = []
        for a in self.agents:
            base = a.score(prompt)
            boost = preference_boost.get(a.name, 0.0)
            scored.append((a.name, round(base + boost, 4)))
        scored.sort(key=lambda x: x[1], reverse=True)

        winner = scored[0][0] if scored and scored[0][1] > 0 else self._heuristic_fallback(prompt)
        winner_score = scored[0][1] if scored else 0.0

        return {
            "agent": winner,
            "score": winner_score,
            "sentiment": sentiment,
            "urgency": signals["urgent"],
            "signals": signals,
            "active_signals": active,
            "all_scored": [(n, s) for n, s in scored],
        }

    def learning_stats(self) -> dict:
        """Return statistics about adaptive learning state.
        Returns feedback count, accuracy, per-agent priority adjustments.
        """
        if not hasattr(self, '_feedback_history'):
            self._feedback_history = []
        total = len(self._feedback_history)
        correct = sum(1 for _, _, ok in self._feedback_history if ok) if total else 0
        priority_changes = {a.name: round(a.priority, 4) for a in self.agents}
        return {
            "total_feedback": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else None,
            "priority_changes": priority_changes,
        }

    def reset_learning(self, priority: float = 1.0) -> dict:
        """Reset all adaptive learning state.
        Clears feedback history and resets agent priorities.
        Returns dict with 'cleared_feedback' count and 'reset_priorities'.
        """
        if not hasattr(self, '_feedback_history'):
            self._feedback_history = []
        cleared = len(self._feedback_history)
        self._feedback_history = []
        reset = {}
        for a in self.agents:
            old = a.priority
            a.priority = priority
            reset[a.name] = {"before": round(old, 4), "after": round(priority, 4)}
        return {"cleared_feedback": cleared, "reset_priorities": reset}

    def detect_language(self, prompt: str) -> str:
        """Detect the dominant language of a prompt.
        Returns ISO 639-1 code: 'zh', 'en', 'ja', 'ko', etc.
        Uses unicode range heuristics — zero dependencies.
        """
        if not prompt.strip():
            return "unknown"
        counts = {"zh": 0, "ja": 0, "ko": 0, "other": 0}
        for ch in prompt:
            cp = ord(ch)
            if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
                counts["zh"] += 1
            elif 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
                counts["ja"] += 1
            elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
                counts["ko"] += 1
            elif cp > 127:
                counts["other"] += 1
        total_cjk = counts["zh"] + counts["ja"] + counts["ko"] + counts["other"]
        if total_cjk == 0:
            return "en"
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else "en"

    def route_by_language(self, prompt: str, lang_map: Optional[dict[str, list[str]]] = None) -> dict:
        """Route a prompt preferring agents that match the detected language.
        lang_map: optional mapping from language code to list of agent names.
        Returns dict with 'language', 'agent', 'score', 'lang_candidates'.
        """
        lang = self.detect_language(prompt)
        default_lang_map = {
            "zh": ["translator", "writer"],
            "ja": ["translator"],
            "ko": ["translator"],
        }
        lmap = lang_map or default_lang_map
        preferred = lmap.get(lang, [])
        agent, score, all_scores = self.route(prompt)
        # Boost preferred agents by 20%
        boosted = []
        for name, s in all_scores:
            if name in preferred:
                boosted.append((name, s * 1.2))
            else:
                boosted.append((name, s))
        boosted.sort(key=lambda x: x[1], reverse=True)
        final_agent = boosted[0][0] if boosted else agent
        final_score = boosted[0][1] if boosted else score
        return {
            "language": lang,
            "agent": final_agent,
            "score": round(final_score, 4),
            "lang_candidates": preferred if preferred else ["any"],
            "all_scores": [(n, round(s, 4)) for n, s in boosted],
        }

    def route_by_complexity(self, prompt: str) -> dict:
        """Estimate prompt complexity and route accordingly.
        Complexity signals: length, multi-sentence, technical terms, nested clauses.
        Returns dict with 'complexity' (low/medium/high), 'agent', 'score', 'signals'.
        """
        signals = {}
        signals["word_count"] = len(prompt.split())
        signals["sentence_count"] = len([s for s in re.split(r'[.!?。！？]', prompt) if s.strip()])
        signals["has_code"] = bool(re.search(r'[<>{}()\[\]=;]', prompt))
        signals["technical_terms"] = len(re.findall(r'\b(API|SDK|HTTP|database|algorithm|architecture|protocol|framework|infrastructure|microservice)\b', prompt, re.I))
        signals["has_negation"] = bool(re.search(r"\b(not|don't|never|no|except|without|avoid)\b", prompt, re.I))

        score = 0
        score += min(signals["word_count"] / 20, 2)  # 0-2
        score += min(signals["sentence_count"] / 3, 2)  # 0-2
        score += 1 if signals["has_code"] else 0
        score += min(signals["technical_terms"], 2)
        score += 0.5 if signals["has_negation"] else 0

        if score < 1.5:
            complexity = "low"
        elif score < 3.0:
            complexity = "medium"
        else:
            complexity = "high"

        agent, s, all_scores = self.route(prompt)
        # High complexity boosts planner/architect types
        if complexity == "high":
            boosted = []
            for name, sc in all_scores:
                boost = 1.15 if name in ("planner", "reviewer") else 1.0
                boosted.append((name, sc * boost))
            boosted.sort(key=lambda x: x[1], reverse=True)
            agent = boosted[0][0]
            s = boosted[0][1]

        return {
            "complexity": complexity,
            "complexity_score": round(score, 2),
            "agent": agent,
            "score": round(s, 4),
            "signals": signals,
        }

    def agent_graph(self) -> dict:
        """Build a similarity/connectivity graph between agents.
        Returns dict with 'nodes' (agent names) and 'edges' (pairs with similarity score).
        """
        names = [a.name for a in self.agents]
        nodes = names
        edges = []
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i < j:
                    sim = self.agent_similarity(a, b)
                    edges.append({
                        "from": a,
                        "to": b,
                        "similarity": round(sim.get("similarity", 0), 4),
                        "shared_keywords": sim.get("shared_keywords", []),
                    })
        return {"nodes": nodes, "edges": edges}

    def export_state(self) -> dict:
        """Export full router state as JSON-serializable dict.
        Includes agents, feedback history, and config.
        """
        agents_data = []
        for a in self.agents:
            agents_data.append({
                "name": a.name,
                "description": a.description,
                "keywords": a.keywords,
                "patterns": a.patterns,
                "priority": a.priority,
            })
        feedback = getattr(self, '_feedback_history', [])
        return {
            "agents": agents_data,
            "feedback": feedback,
            "agent_count": len(agents_data),
            "version": "1.0",
        }

    def import_state(self, data: dict) -> dict:
        """Import router state from a dict (as produced by export_state).
        Replaces current agents and feedback. Returns summary.
        """
        self.agents = []
        for ad in data.get("agents", []):
            self.agents.append(Agent(
                name=ad["name"],
                description=ad["description"],
                keywords=ad.get("keywords", []),
                patterns=ad.get("patterns", []),
                priority=ad.get("priority", 1.0),
            ))
        self._feedback_history = data.get("feedback", [])
        return {
            "imported_agents": len(self.agents),
            "imported_feedback": len(self._feedback_history),
        }

    def route_by_length(self, prompt: str, length_map: Optional[dict[str, list[str]]] = None) -> dict:
        """Route based on prompt length (word count).
        length_map maps categories to agent name lists:
          {"short": ["fast-agent"], "medium": [...], "long": ["deep-agent"]}
        Default categories: short (<8), medium (8-20), long (>20).
        Falls back to normal route() if no mapping or no matching agent found.
        """
        word_count = len(prompt.split())
        if word_count < 8:
            category = "short"
        elif word_count <= 20:
            category = "medium"
        else:
            category = "long"

        agent, score, all_scores = self.route(prompt)

        if length_map and category in length_map:
            preferred = [n for n in length_map[category] if any(a.name == n for a in self.agents)]
            if preferred:
                preferred_scores = [(n, s) for n, s in all_scores if n in preferred]
                if preferred_scores:
                    agent = preferred_scores[0][0]
                    score = preferred_scores[0][1]

        return {
            "word_count": word_count,
            "category": category,
            "agent": agent,
            "score": round(score, 4),
            "all_scores": [(n, round(s, 4)) for n, s in all_scores],
        }

    def prune_agents(self, min_score: float = 0.1) -> dict:
        """Remove agents whose max possible score (sum of keyword weights) is below threshold.
        Returns dict with removed agent names and count.
        """
        removed = []
        kept = []
        for agent in self.agents:
            max_kw_weight = sum(len(kw.split()) for kw in agent.keywords)
            if max_kw_weight < min_score and len(agent.keywords) == 0:
                removed.append(agent.name)
            else:
                kept.append(agent)
        self.agents = kept
        return {
            "removed": removed,
            "removed_count": len(removed),
            "remaining": len(self.agents),
        }

    def optimize_weights(self, feedback_log: list[tuple[str, str, bool]]) -> dict:
        """Auto-tune agent keyword weights from historical feedback.
        feedback_log: list of (prompt, correct_agent, was_correct) tuples.
        For misrouted prompts, boosts correct agent's matching keywords and
        reduces incorrect agent's matching keywords.
        Returns summary of adjustments made.
        """
        adjustments = []
        agent_map = {a.name: a for a in self.agents}

        for prompt, correct, was_ok in feedback_log:
            if correct not in agent_map:
                continue
            words = set(prompt.lower().split())
            correct_agent = agent_map[correct]

            if not was_ok:
                # Boost correct agent keywords that match prompt words
                boosted = []
                for kw in correct_agent.keywords:
                    if any(w in kw.lower() for w in words):
                        boosted.append(kw)
                if boosted:
                    # Add missing keywords from prompt words
                    for w in words:
                        if len(w) > 3 and w not in correct_agent.keywords:
                            correct_agent.keywords.append(w)
                    adjustments.append({"correct": correct, "boosted_keywords": boosted})

        return {
            "processed": len(feedback_log),
            "adjustments": adjustments,
            "adjustment_count": len(adjustments),
            "agents_affected": list(set(a["correct"] for a in adjustments)),
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
