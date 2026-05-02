#!/usr/bin/env python3
"""Quick sanity tests for prompt_router."""
from prompt_router import PromptRouter

router = PromptRouter()

tests = [
    ("Fix the authentication bug in login.py", "coder"),
    ("Summarize the Q4 revenue report", "researcher"),
    ("写一封英文邮件给客户", "translator"),
    ("Review the security of this API", "reviewer"),
    ("Plan the architecture for a new microservice", "planner"),
    ("Write a blog post about Rust", "writer"),
    ("How does quantum computing work?", "researcher"),
    ("Create a Python script to backup files", "coder"),
]

passed = 0
for prompt, expected in tests:
    agent, score, _ = router.route(prompt)
    ok = agent == expected
    status = "✅" if ok else "❌"
    if ok:
        passed += 1
    else:
        print(f"  {status} '{prompt}' → {agent} (expected {expected}, score={score:.2f})")

print(f"\n{passed}/{len(tests)} tests passed")

# --- explain() tests ---
explain_tests_passed = 0
explain_tests_total = 0

# Test 1: explain returns structured dict with correct keys
explain_tests_total += 1
result = router.explain("Fix the login bug")
assert isinstance(result, dict), "explain should return dict"
assert result["best_agent"] == "coder", f"expected coder, got {result['best_agent']}"
assert "agents" in result and "reasons" in result["agents"]["coder"], "missing reasons"
assert len(result["agents"]["coder"]["reasons"]) > 0, "coder should have match reasons"
print("  ✅ explain() returns structured dict with reasons")
explain_tests_passed += 1

# Test 2: explain shows keyword matches for researcher
explain_tests_total += 1
result = router.explain("Summarize the Q4 revenue report")
assert result["best_agent"] == "researcher"
reasons = result["agents"]["researcher"]["reasons"]
assert any("keyword" in r for r in reasons), "researcher should have keyword matches"
print("  ✅ explain() shows keyword matches for researcher")
explain_tests_passed += 1

# Test 3: explain shows pattern matches for coder
explain_tests_total += 1
result = router.explain("Fix the authentication bug in login.py")
coder_reasons = result["agents"]["coder"]["reasons"]
assert any("pattern" in r for r in coder_reasons), "coder should have pattern matches (file.ext)"
print("  ✅ explain() shows pattern matches for coder")
explain_tests_passed += 1

# Test 4: explain on empty/gibberish prompt uses fallback
explain_tests_total += 1
result = router.explain("xyzzy foo bar")
assert result["best_agent"] in [a.name for a in router.agents], "fallback should return valid agent"
print("  ✅ explain() handles unknown prompts with fallback")
explain_tests_passed += 1

print(f"\n{explain_tests_passed}/{explain_tests_total} explain tests passed")

# --- route_with_confidence() tests ---
conf_tests_passed = 0
conf_tests_total = 0

# Test 1: high-confidence prompt returns confident
conf_tests_total += 1
agent, score, status = router.route_with_confidence("Fix the authentication bug in login.py")
assert status == "confident", f"expected confident, got {status}"
assert agent == "coder"
print("  ✅ route_with_confidence: clear prompt → confident")
conf_tests_passed += 1

# Test 2: gibberish returns no_match
conf_tests_total += 1
agent, score, status = router.route_with_confidence("xyzzy plugh")
assert status == "no_match", f"expected no_match, got {status}"
assert agent is None
print("  ✅ route_with_confidence: gibberish → no_match")
conf_tests_passed += 1

# Test 3: custom threshold works
conf_tests_total += 1
agent, score, status = router.route_with_confidence("Fix the authentication bug in login.py", threshold=999.0)
assert status == "low_confidence", f"expected low_confidence, got {status}"
print("  ✅ route_with_confidence: high threshold → low_confidence")
conf_tests_passed += 1

print(f"\n{conf_tests_passed}/{conf_tests_total} confidence tests passed")

# --- route_batch() tests ---
batch_passed = 0
batch_total = 0

# Test 1: basic batch routing
batch_total += 1
prompts = [
    "Fix the login bug",
    "Summarize the Q4 report",
    "Write a blog post about AI",
]
batch_result = router.route_batch(prompts)
assert "results" in batch_result and "stats" in batch_result
assert len(batch_result["results"]) == 3
assert batch_result["stats"]["total"] == 3
print("  ✅ route_batch: returns results + stats for 3 prompts")
batch_passed += 1

# Test 2: distribution is correct
batch_total += 1
dist = batch_result["stats"]["distribution"]
assert "coder" in dist, "coder should be in distribution"
assert sum(dist.values()) == 3, "distribution should sum to total"
print("  ✅ route_batch: distribution sums to total")
batch_passed += 1

# Test 3: empty batch
batch_total += 1
empty = router.route_batch([])
assert empty["stats"]["total"] == 0
assert empty["stats"]["avg_score"] == 0
print("  ✅ route_batch: handles empty list")
batch_passed += 1

# Test 4: status_counts populated
batch_total += 1
sc = batch_result["stats"]["status_counts"]
assert "confident" in sc or "no_match" in sc, "status_counts should have entries"
print("  ✅ route_batch: status_counts populated")
batch_passed += 1

print(f"\n{batch_passed}/{batch_total} batch tests passed")

# --- route_with_fallback() tests ---
fb_passed = 0
fb_total = 0

# Test 1: high-confidence → no fallback needed
fb_total += 1
result = router.route_with_fallback("Fix the authentication bug in login.py")
assert result["agent"] == "coder", f"expected coder, got {result['agent']}"
assert result["fallback_used"] is False
assert len(result["chain"]) == 1
print("  ✅ route_with_fallback: high-confidence → primary agent, no fallback")
fb_passed += 1

# Test 2: chain contains scored agents and is sorted
fb_total += 1
result = router.route_with_fallback("Plan the architecture for a new system", threshold=999.0)
assert len(result["chain"]) == len(router.agents), f"chain should have all {len(router.agents)} agents"
assert result["chain"][0]["score"] >= result["chain"][-1]["score"]
print("  ✅ route_with_fallback: chain sorted by score descending")
fb_passed += 1

# Test 3: very high threshold forces fallback to best available
fb_total += 1
result = router.route_with_fallback("Fix the login bug", threshold=999.0)
assert result["fallback_used"] is True
assert result["agent"] is not None
print("  ✅ route_with_fallback: unmeetable threshold → fallback to best")
fb_passed += 1

print(f"\n{fb_passed}/{fb_total} fallback tests passed")

# --- add_agent / remove_agent / list_agents tests ---
agent_mgmt_passed = 0
agent_mgmt_total = 0

# Test 1: add_agent adds and routes to new agent
agent_mgmt_total += 1
custom = PromptRouter()
from prompt_router import Agent
custom.add_agent(Agent("devops", "Deploy and manage infrastructure",
                        keywords=["deploy", "kubernetes", "docker", "ci/cd", "infrastructure"]))
agent, score, _ = custom.route("Deploy the new docker container to kubernetes")
assert agent == "devops", f"expected devops, got {agent}"
assert score > 0
print("  ✅ add_agent: new agent routes correctly")
agent_mgmt_passed += 1

# Test 2: add_agent no-op on duplicate name
agent_mgmt_total += 1
count_before = len(custom.agents)
custom.add_agent(Agent("devops", "duplicate", keywords=["x"]))
assert len(custom.agents) == count_before, "duplicate should be ignored"
print("  ✅ add_agent: duplicate name ignored")
agent_mgmt_passed += 1

# Test 3: remove_agent works
agent_mgmt_total += 1
custom2 = PromptRouter()
orig_count = len(custom2.agents)
removed = custom2.remove_agent("writer")
assert removed is True
assert len(custom2.agents) == orig_count - 1
assert all(a.name != "writer" for a in custom2.agents)
print("  ✅ remove_agent: removes agent and returns True")
agent_mgmt_passed += 1

# Test 4: remove_agent returns False for missing name
agent_mgmt_total += 1
assert custom2.remove_agent("nonexistent") is False
print("  ✅ remove_agent: returns False for missing agent")
agent_mgmt_passed += 1

# Test 5: list_agents returns summaries
agent_mgmt_total += 1
listing = router.list_agents()
assert isinstance(listing, list) and len(listing) >= 6
assert all("name" in item and "keywords" in item for item in listing)
print("  ✅ list_agents: returns structured summaries")
agent_mgmt_passed += 1

print(f"\n{agent_mgmt_passed}/{agent_mgmt_total} agent management tests passed")

# --- save_config / load_config tests ---
import os, tempfile
config_passed = 0
config_total = 0

# Test 1: save+load round-trip preserves agents
config_total += 1
r = PromptRouter()
tmpf = os.path.join(tempfile.gettempdir(), "test_router_config.json")
r.save_config(tmpf)
r2 = PromptRouter([Agent("x", "y")])  # empty-ish
assert len(r2.agents) == 1
r2.load_config(tmpf)
assert len(r2.agents) == len(r.agents)
assert r2.agents[0].name == r.agents[0].name
print("  ✅ save_config/load_config: round-trip preserves agents")
config_passed += 1

# Test 2: loaded config routes identically
config_total += 1
prompt = "Fix the login bug"
a1, s1, _ = r.route(prompt)
a2, s2, _ = r2.route(prompt)
assert a1 == a2 and abs(s1 - s2) < 0.001, f"{a1}/{s1} vs {a2}/{s2}"
print("  ✅ load_config: loaded router routes identically")
config_passed += 1

# Test 3: custom agent survives round-trip
config_total += 1
custom_r = PromptRouter()
custom_r.add_agent(Agent("custom", "test", keywords=["flibble"]))
custom_r.save_config(tmpf)
custom_r2 = PromptRouter()
custom_r2.load_config(tmpf)
agent, score, _ = custom_r2.route("flibble the thing")
assert agent == "custom", f"expected custom, got {agent}"
print("  ✅ save/load: custom agent survives round-trip")
config_passed += 1

os.unlink(tmpf)
print(f"\n{config_passed}/{config_total} config tests passed")

# --- route_top_k tests ---
topk_passed = 0
topk_total = 0

# Test 1: returns k results
topk_total += 1
results = router.route_top_k("Fix the login bug", k=3)
assert len(results) == 3
assert results[0]["agent"] == "coder"
print("  ✅ route_top_k: returns top 3 agents, coder first")
topk_passed += 1

# Test 2: k larger than agents returns all
topk_total += 1
big = router.route_top_k("test", k=100)
assert len(big) == len(router.agents)
print("  ✅ route_top_k: k > agents returns all agents")
topk_passed += 1

# Test 3: each result has reasons
topk_total += 1
top3 = router.route_top_k("Summarize the revenue report", k=2)
assert all("reasons" in r for r in top3)
assert len(top3[0]["reasons"]) > 0, "top result should have reasons"
print("  ✅ route_top_k: results include reasons")
topk_passed += 1

# Test 4: sorted descending by score
topk_total += 1
top = router.route_top_k("Deploy to kubernetes", k=5)
scores = [r["score"] for r in top]
assert scores == sorted(scores, reverse=True)
print("  ✅ route_top_k: sorted descending by score")
topk_passed += 1

print(f"\n{topk_passed}/{topk_total} top_k tests passed")

total_passed = passed + explain_tests_passed + conf_tests_passed + batch_passed + fb_passed + agent_mgmt_passed + config_passed + topk_passed
total_tests = len(tests) + explain_tests_total + conf_tests_total + batch_total + fb_total + agent_mgmt_total + config_total + topk_total
# --- route_ensemble tests ---
ens_passed = 0
ens_total = 0

# Test 1: returns k agents with weights
ens_total += 1
result = router.route_ensemble("Fix the login bug", k=3)
assert "agents" in result and "total_weight" in result
assert len(result["agents"]) == 3
assert result["agents"][0]["agent"] == "coder"
assert all("weight" in a for a in result["agents"])
print("  ✅ route_ensemble: returns 3 agents with weights, coder first")
ens_passed += 1

# Test 2: weights sum to ~1.0
ens_total += 1
wsum = sum(a["weight"] for a in result["agents"])
assert abs(wsum - 1.0) < 0.01, f"weights should sum to ~1.0, got {wsum}"
print("  ✅ route_ensemble: weights sum to ~1.0")
ens_passed += 1

# Test 3: custom weights bias results
ens_total += 1
biased = router.route_ensemble("Fix the login bug", k=3, weights={"coder": 2.0, "researcher": 0.5})
coder_w = next(a["weight"] for a in biased["agents"] if a["agent"] == "coder")
researcher_w = next((a["weight"] for a in biased["agents"] if a["agent"] == "researcher"), 0)
assert coder_w > researcher_w, "coder should have higher weight with bias"
print("  ✅ route_ensemble: custom weights bias distribution")
ens_passed += 1

# Test 4: each agent has reasons
ens_total += 1
assert all(len(a["reasons"]) >= 0 for a in result["agents"]), "each agent should have reasons"
assert len(result["agents"][0]["reasons"]) > 0, "top agent should have reasons"
print("  ✅ route_ensemble: agents include reasons")
ens_passed += 1

# Test 5: k=1 returns single agent
ens_total += 1
single = router.route_ensemble("Deploy to kubernetes", k=1)
assert len(single["agents"]) == 1
assert single["agents"][0]["weight"] == 1.0
print("  ✅ route_ensemble: k=1 returns single agent with weight 1.0")
ens_passed += 1

print(f"\n{ens_passed}/{ens_total} ensemble tests passed")

# --- merge_routers tests ---
merge_passed = 0
merge_total = 0

# Test 1: merge two routers deduplicates by name
merge_total += 1
r1 = PromptRouter([Agent("a", "first", keywords=["x"]), Agent("b", "second", keywords=["y"])])
r2 = PromptRouter([Agent("b", "duplicate", keywords=["z"]), Agent("c", "third", keywords=["w"])])
merged = PromptRouter.merge_routers(r1, r2)
names = [a.name for a in merged.agents]
assert names == ["a", "b", "c"], f"expected [a,b,c], got {names}"
assert merged.agents[1].description == "second", "first occurrence wins"
print("  ✅ merge_routers: deduplicates by name, first wins")
merge_passed += 1

# Test 2: merged router routes correctly
merge_total += 1
agent, score, _ = merged.route("x thing")
assert agent == "a", f"expected a, got {agent}"
print("  ✅ merge_routers: merged router routes correctly")
merge_passed += 1

# Test 3: merge single router returns copy
merge_total += 1
single_merged = PromptRouter.merge_routers(r1)
assert len(single_merged.agents) == len(r1.agents)
assert single_merged.agents is not r1.agents, "should be new list"
print("  ✅ merge_routers: single router returns new copy")
merge_passed += 1

# Test 4: merge empty is valid
merge_total += 1
empty_merged = PromptRouter.merge_routers()
assert len(empty_merged.agents) == 0
print("  ✅ merge_routers: no routers → empty router")
merge_passed += 1

print(f"\n{merge_passed}/{merge_total} merge tests passed")

# --- route_adaptive tests ---
adaptive_passed = 0
adaptive_total = 0

# Test 1: routes correctly without feedback
adaptive_total += 1
result = router.route_adaptive("Fix the login bug")
assert result["agent"] == "coder"
assert result["feedback"] is None
assert result["history_size"] == 0
print("  ✅ route_adaptive: routes without feedback")
adaptive_passed += 1

# Test 2: correct feedback recorded
adaptive_total += 1
r = PromptRouter()  # fresh router
result = r.route_adaptive("Fix the login bug", correct_agent="coder")
assert result["feedback"] == "correct"
assert result["history_size"] == 1
assert result["accuracy"] == 1.0
print("  ✅ route_adaptive: correct feedback recorded")
adaptive_passed += 1

# Test 3: wrong feedback adjusts priorities
adaptive_total += 1
r2 = PromptRouter()
result = r2.route_adaptive("Fix the login bug", correct_agent="researcher")
assert result["feedback"] == "corrected"
assert result["accuracy"] == 0.0
# Check researcher priority boosted
res_priority = next(a.priority for a in r2.agents if a.name == "researcher")
coder_priority = next(a.priority for a in r2.agents if a.name == "coder")
assert res_priority > 1.0, "researcher priority should increase"
assert coder_priority < 1.0, "coder priority should decrease"
print("  ✅ route_adaptive: wrong feedback adjusts priorities")
adaptive_passed += 1

# Test 4: accuracy tracks over multiple calls
adaptive_total += 1
r3 = PromptRouter()
r3.route_adaptive("Fix bug", correct_agent="coder")  # correct (coder routes coder)
r3.route_adaptive("Write blog", correct_agent="researcher")  # wrong (writer routes, not researcher)
last = r3.route_adaptive("Summarize data", correct_agent="researcher")  # likely correct
assert last["history_size"] == 3
assert last["accuracy"] is not None and 0 < last["accuracy"] <= 1.0
print("  ✅ route_adaptive: accuracy tracks over multiple calls")
adaptive_passed += 1

# Test 5: history_size without feedback stays same
adaptive_total += 1
r4 = PromptRouter()
r4.route_adaptive("Fix", correct_agent="coder")
s1 = r4.route_adaptive("Test")["history_size"]
assert s1 == 1, "no feedback should not add to history"
print("  ✅ route_adaptive: no feedback = no history entry")
adaptive_passed += 1

print(f"\n{adaptive_passed}/{adaptive_total} adaptive tests passed")

total_passed = passed + explain_tests_passed + conf_tests_passed + batch_passed + fb_passed + agent_mgmt_passed + config_passed + topk_passed + ens_passed + merge_passed + adaptive_passed
total_tests = len(tests) + explain_tests_total + conf_tests_total + batch_total + fb_total + agent_mgmt_total + config_total + topk_total + ens_total + merge_total + adaptive_total
print(f"\n📊 Total: {total_passed}/{total_tests} tests passed")
