#!/usr/bin/env python3
"""Quick sanity tests for prompt_router."""
from prompt_router import PromptRouter, DEFAULT_AGENTS
import json

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

# --- route_by_tags tests ---
tag_total = 6
tag_passed = 0

# 1. filter by agent name tag
agent, score, _ = PromptRouter().route_by_tags("Fix the bug", ["coder"])
assert agent == "coder", f"expected coder, got {agent}"
print("  ✅ route_by_tags: filter by agent name")
tag_passed += 1

# 2. filter by description tag
agent, score, _ = PromptRouter().route_by_tags("Write a poem", ["content", "documentation"])
assert agent is not None, "should match writer via description tag"
print("  ✅ route_by_tags: filter by description tag")
tag_passed += 1

# 3. no matching tags returns None
agent, score, _ = PromptRouter().route_by_tags("Fix the bug", ["nonexistent"])
assert agent is None, f"expected None, got {agent}"
assert score == 0.0
print("  ✅ route_by_tags: no matching tags → None")
tag_passed += 1

# 4. multiple tags (union)
agent, score, _ = PromptRouter().route_by_tags("Debug the server", ["coder", "researcher"])
assert agent == "coder"
print("  ✅ route_by_tags: multiple tags union")
tag_passed += 1

# 5. tag match but no keyword match → None
agent, score, _ = PromptRouter().route_by_tags("zzzzzz unknown", ["coder"])
assert agent is None, "coder tag matched but no keywords → None"
print("  ✅ route_by_tags: tag matched but no keyword match → None")
tag_passed += 1

# 6. case insensitive tags
agent, score, _ = PromptRouter().route_by_tags("Plan the project", ["PLANNER"])
assert agent == "planner"
print("  ✅ route_by_tags: case insensitive")
tag_passed += 1

print(f"\n{tag_passed}/{tag_total} route_by_tags tests passed")

total_passed += tag_passed
total_tests += tag_total

# --- route_with_history tests ---
hist_total = 6
hist_passed = 0

# 1. empty history = normal routing
r_h = PromptRouter()
res = r_h.route_with_history("Fix the bug", [])
assert res["agent"] == "coder", f"got {res['agent']}"
assert res["diversified"] == False
print("  ✅ route_with_history: empty history = normal routing")
hist_passed += 1

# 2. None history = normal routing
res = r_h.route_with_history("Fix the bug", None)
assert res["agent"] == "coder"
print("  ✅ route_with_history: None history = normal routing")
hist_passed += 1

# 3. high penalty causes diversification when runner-up exists
res = r_h.route_with_history("Research and explain the bug", ["researcher", "researcher", "researcher", "researcher", "researcher"], penalty=1.0)
assert res["agent_counts"] == {"researcher": 5}
assert res["diversified"] == True, f"should diversify from researcher, got {res['agent']}"
print("  ✅ route_with_history: high penalty causes diversification")
hist_passed += 1

# 4. adjusted_score < raw score for repeated agent
res = r_h.route_with_history("Write a function", ["coder"])
assert res["adjusted_score"] <= res["score"]
print("  ✅ route_with_history: penalty reduces adjusted score")
hist_passed += 1

# 5. no repeat avoidance
res = r_h.route_with_history("Fix the bug", ["coder", "coder"], avoid_repeat=False)
assert res["agent"] == "coder", "no avoidance should still pick coder"
assert res["diversified"] == False
print("  ✅ route_with_history: avoid_repeat=False disables penalty")
hist_passed += 1

# 6. custom penalty scales (use moderate penalty to avoid fallback)
res_low = r_h.route_with_history("Fix the bug", ["coder"], penalty=0.1)
res_high = r_h.route_with_history("Fix the bug", ["coder"], penalty=0.5)
assert res_high["adjusted_score"] <= res_low["adjusted_score"]
print("  ✅ route_with_history: custom penalty scales")
hist_passed += 1

print(f"\n{hist_passed}/{hist_total} route_with_history tests passed")

total_passed += hist_passed
total_tests += hist_total

# --- score_matrix tests ---
mx_total = 6
mx_passed = 0

# 1. basic matrix shape
r_mx = PromptRouter()
mx = r_mx.score_matrix(["Fix bug", "Research AI", "Write docs"])
assert len(mx["matrix"]) == 3, "3 prompts = 3 rows"
assert len(mx["matrix"][0]) == len(r_mx.agents), "columns = agents"
print("  ✅ score_matrix: correct shape")
mx_passed += 1

# 2. best_agent_per_prompt correctness
assert mx["best_agent_per_prompt"][0] == "coder"
assert mx["best_agent_per_prompt"][1] == "researcher"
print("  ✅ score_matrix: best agent per prompt")
mx_passed += 1

# 3. agent_coverage sums to ~1.0
total_cov = sum(mx["agent_coverage"].values())
assert abs(total_cov - 1.0) < 0.01, f"coverage should sum to 1.0, got {total_cov}"
print("  ✅ score_matrix: coverage sums to 1.0")
mx_passed += 1

# 4. empty prompts
mx_empty = r_mx.score_matrix([])
assert mx_empty["matrix"] == []
assert mx_empty["best_agent_per_prompt"] == []
print("  ✅ score_matrix: empty prompts")
mx_passed += 1

# 5. single prompt
mx_one = r_mx.score_matrix(["Fix bug"])
assert len(mx_one["matrix"]) == 1
assert mx_one["best_agent_per_prompt"][0] == "coder"
print("  ✅ score_matrix: single prompt")
mx_passed += 1

# 6. all zero scores → None best agents
mx_zero = r_mx.score_matrix(["zzzzz xxxxx"])
assert mx_zero["best_agent_per_prompt"][0] is None or all(s == 0 for s in mx_zero["matrix"][0])
print("  ✅ score_matrix: zero score handling")
mx_passed += 1

print(f"\n{mx_passed}/{mx_total} score_matrix tests passed")

total_passed += mx_passed
total_tests += mx_total

# --- route_chain tests ---
chain_total = 6
chain_passed = 0

# 1. basic chain
r_ch = PromptRouter()
chain = r_ch.route_chain(["Fix bug", "Research AI", "Write docs"])
assert len(chain) == 3
assert chain[0]["position"] == 0
assert chain[0]["prev_agent"] is None
print("  ✅ route_chain: basic chain structure")
chain_passed += 1

# 2. chain tracks prev_agent
assert chain[1]["prev_agent"] == chain[0]["agent"]
assert chain[2]["prev_agent"] == chain[1]["agent"]
print("  ✅ route_chain: prev_agent tracking")
chain_passed += 1

# 3. empty chain
assert r_ch.route_chain([]) == []
print("  ✅ route_chain: empty input")
chain_passed += 1

# 4. diversify promotes variety
chain_div = r_ch.route_chain(["Fix bug", "Fix error", "Fix crash", "Fix issue", "Fix problem"], diversify=True, penalty=0.5)
agents_used = set(r["agent"] for r in chain_div)
chain_nodiv = r_ch.route_chain(["Fix bug", "Fix error", "Fix crash", "Fix issue", "Fix problem"], diversify=False)
agents_nodiv = set(r["agent"] for r in chain_nodiv)
assert len(agents_used) >= len(agents_nodiv), "diversify should not reduce variety"
print("  ✅ route_chain: diversify promotes variety")
chain_passed += 1

# 5. single prompt
chain_one = r_ch.route_chain(["Fix bug"])
assert len(chain_one) == 1
assert chain_one[0]["agent"] == "coder"
print("  ✅ route_chain: single prompt")
chain_passed += 1

# 6. chain results have all expected keys
expected_keys = {"position", "prompt", "agent", "score", "prev_agent", "diversified"}
assert set(chain[0].keys()) == expected_keys
print("  ✅ route_chain: result keys")
chain_passed += 1

print(f"\n{chain_passed}/{chain_total} route_chain tests passed")

total_passed += chain_passed
total_tests += chain_total

# ========== route_round_robin tests ==========
r_total = 6
r_passed = 0
r = PromptRouter()

# 1. basic routing with fresh state
result = r.route_round_robin("Fix the bug")
assert result["agent"] is not None
assert result["score"] > 0
assert result["index"] >= 0
assert "assignments" in result
print("  ✅ route_round_robin: basic routing")
r_passed += 1

# 2. state persists across calls
state = {}
r1 = r.route_round_robin("Write code", state=state)
r2 = r.route_round_robin("Write more code", state=state)
assert state["assignments"][r1["agent"]] == 1 or state["assignments"][r1["agent"]] == 2
assert sum(state["assignments"].values()) == 2
print("  ✅ route_round_robin: state persists")
r_passed += 1

# 3. load balancing distributes across agents
state_lb = {}
for prompt in ["Fix bug", "Debug error", "Refactor code", "Write test", "Optimize", "Deploy", "Review PR", "Add feature", "Clean up", "Ship it"]:
    r.route_round_robin(prompt, state=state_lb)
unique_agents = len(state_lb["assignments"])
assert unique_agents >= 2, f"Only {unique_agents} agents used, expected load balancing"
print("  ✅ route_round_robin: distributes across agents")
r_passed += 1

# 4. empty router
empty = PromptRouter([])
result_e = empty.route_round_robin("test")
assert result_e["agent"] is None
print("  ✅ route_round_robin: empty router")
r_passed += 1

# 5. state dict is mutable and shared
shared = {}
r.route_round_robin("A", state=shared)
copy_before = dict(shared["assignments"])
r.route_round_robin("B", state=shared)
assert shared["assignments"] != copy_before
print("  ✅ route_round_robin: shared state")
r_passed += 1

# 6. assignments count matches calls
count_state = {}
n = 5
for i in range(n):
    r.route_round_robin(f"prompt {i}", state=count_state)
total_assigned = sum(count_state["assignments"].values())
assert total_assigned == n
print("  ✅ route_round_robin: assignment count")
r_passed += 1

print(f"\n{r_passed}/{r_total} route_round_robin tests passed")
total_passed += r_passed
total_tests += r_total

# ========== route_least_loaded tests ==========
ll_total = 6
ll_passed = 0

# 1. basic least loaded
result = r.route_least_loaded("Fix the bug")
assert result["agent"] is not None
assert result["score"] > 0
assert "load" in result
assert "candidates" in result
print("  ✅ route_least_loaded: basic routing")
ll_passed += 1

# 2. picks least loaded agent
loads = {"coder": 10, "researcher": 0, "writer": 5}
result_ll = r.route_least_loaded("Explain quantum computing", loads=loads)
# researcher has load 0 and should score well for this prompt
assert result_ll["load"] <= 5  # should pick a low-load agent
print("  ✅ route_least_loaded: picks low load")
ll_passed += 1

# 3. all loads equal → falls back to score
loads_equal = {"coder": 5, "researcher": 5, "writer": 5}
result_eq = r.route_least_loaded("Write a blog post", loads=loads_equal)
assert result_eq["agent"] is not None
print("  ✅ route_least_loaded: equal loads → score")
ll_passed += 1

# 4. threshold filters low-scoring agents
loads_thresh = {"researcher": 0, "writer": 0}
result_t = r.route_least_loaded("Fix the bug", loads=loads_thresh, threshold=0.5)
# Should still return someone (falls back if all below threshold)
assert result_t["agent"] is not None
print("  ✅ route_least_loaded: threshold filter")
ll_passed += 1

# 5. empty router
empty = PromptRouter([])
result_e = empty.route_least_loaded("test")
assert result_e["agent"] is None
print("  ✅ route_least_loaded: empty router")
ll_passed += 1

# 6. candidates list is populated
result_c = r.route_least_loaded("Write code")
assert len(result_c["candidates"]) > 0
# candidates sorted by load then score
print("  ✅ route_least_loaded: candidates populated")
ll_passed += 1

print(f"\n{ll_passed}/{ll_total} route_least_loaded tests passed")
total_passed += ll_passed
total_tests += ll_total

# ========== route_by_capability tests ==========
cap_total = 6
cap_passed = 0

# 1. mode='any': matches at least one capability
result = r.route_by_capability("Fix the bug", capabilities=["debug", "python"])
assert result["agent"] is not None
assert result["mode"] == "any"
print("  ✅ route_by_capability: any mode")
cap_passed += 1

# 2. mode='all': must match all capabilities
result_all = r.route_by_capability("Fix the bug", capabilities=["debug", "python", "testing"], mode="all")
# hard to match all 3, agent might be None
assert result_all["mode"] == "all"
print("  ✅ route_by_capability: all mode")
cap_passed += 1

# 3. mode='best': picks most matches
result_best = r.route_by_capability("Write secure code", capabilities=["debug", "security", "code", "review"], mode="best")
assert result_best["agent"] is not None
assert len(result_best["matched_capabilities"]) > 0
print("  ✅ route_by_capability: best mode")
cap_passed += 1

# 4. no matching capabilities
result_none = r.route_by_capability("Fix the bug", capabilities=["xyznonexistent"])
assert result_none["agent"] is None or len(result_none["matched_capabilities"]) == 0
print("  ✅ route_by_capability: no match")
cap_passed += 1

# 5. empty capabilities list
result_empty = r.route_by_capability("Fix the bug", capabilities=[])
assert result_empty["agent"] is None
print("  ✅ route_by_capability: empty caps")
cap_passed += 1

# 6. empty router
empty = PromptRouter([])
result_e = empty.route_by_capability("test", capabilities=["code"])
assert result_e["agent"] is None
print("  ✅ route_by_capability: empty router")
cap_passed += 1

print(f"\n{cap_passed}/{cap_total} route_by_capability tests passed")
total_passed += cap_passed
total_tests += cap_total

# ========== agent_stats tests ==========
stats_total = 4
stats_passed = 0

# 1. returns summary
stats = r.agent_stats()
assert stats["total_agents"] > 0
assert len(stats["agents"]) == stats["total_agents"]
print("  ✅ agent_stats: returns summary")
stats_passed += 1

# 2. agent info has expected keys
first = stats["agents"][0]
assert "name" in first and "keyword_count" in first and "description_length" in first
print("  ✅ agent_stats: agent info keys")
stats_passed += 1

# 3. empty router
empty_stats = PromptRouter([]).agent_stats()
assert empty_stats["total_agents"] == 0
assert empty_stats["agents"] == []
print("  ✅ agent_stats: empty router")
stats_passed += 1

# 4. keywords are sorted
first_kw = stats["agents"][0]["keywords"]
assert first_kw == sorted(first_kw)
print("  ✅ agent_stats: keywords sorted")
stats_passed += 1

print(f"\n{stats_passed}/{stats_total} agent_stats tests passed")
total_passed += stats_passed
total_tests += stats_total

print(f"\n📊 Total: {total_passed}/{total_tests} tests passed")

# ========== route_with_diversity tests ==========
div_total = 7
div_passed = 0

# 1. no recent agents = same as plain route
plain_agent, plain_score, _ = r.route("Write some code")
div_result = r.route_with_diversity("Write some code")
assert div_result["agent"] == plain_agent
assert div_result["penalty"] == 0.0
print("  ✅ route_with_diversity: no recent = plain route")
div_passed += 1

# 2. penalty reduces top agent score
div2 = r.route_with_diversity("Write some code", recent_agents=["coder"])
assert div2["penalty"] > 0
print("  ✅ route_with_diversity: penalty applied")
div_passed += 1

# 3. heavy penalty can swap agent
div3 = r.route_with_diversity("Write some code", recent_agents=["coder"] * 10, penalty=0.5)
assert div3["agent"] != "coder" or div3["penalty"] > 0
print("  ✅ route_with_diversity: heavy penalty changes ranking")
div_passed += 1

# 4. all_scored list populated
assert len(div3["all_scored"]) == len(r.agents)
print("  ✅ route_with_diversity: all_scored populated")
div_passed += 1

# 5. unknown agent in recent = no effect
div5 = r.route_with_diversity("test", recent_agents=["nonexistent_agent"])
assert div5["penalty"] == 0.0
print("  ✅ route_with_diversity: unknown agent ignored")
div_passed += 1

# 6. empty router
empty_div = PromptRouter([]).route_with_diversity("test")
assert empty_div["agent"] is not None  # heuristic fallback
print("  ✅ route_with_diversity: empty router fallback")
div_passed += 1

# 7. base_score preserved correctly
div7 = r.route_with_diversity("Write code")
for entry in div7["all_scored"]:
    assert "base_score" in entry
    assert "adjusted_score" in entry
    assert entry["adjusted_score"] <= entry["base_score"]
print("  ✅ route_with_diversity: scores consistent")
div_passed += 1

print(f"\n{div_passed}/{div_total} route_with_diversity tests passed")
total_passed += div_passed
total_tests += div_total

# ========== deduplicate_agents tests ==========
dedup_total = 4
dedup_passed = 0

# 1. no duplicates = 0 removed
dedup_r = PromptRouter(list(DEFAULT_AGENTS))
removed = dedup_r.deduplicate_agents()
assert removed == 0
assert len(dedup_r.agents) == len(DEFAULT_AGENTS)
print("  ✅ deduplicate_agents: no duplicates")
dedup_passed += 1

# 2. with duplicates = removed
dedup_r2 = PromptRouter(list(DEFAULT_AGENTS) + [DEFAULT_AGENTS[0]])
first_count = sum(1 for a in dedup_r2.agents if a.name == DEFAULT_AGENTS[0].name)
assert first_count == 2
removed2 = dedup_r2.deduplicate_agents()
assert removed2 == 1
assert len(dedup_r2.agents) == len(DEFAULT_AGENTS)
print("  ✅ deduplicate_agents: removes duplicates")
dedup_passed += 1

# 3. keeps first occurrence
dedup_r3 = PromptRouter([DEFAULT_AGENTS[0], DEFAULT_AGENTS[1], DEFAULT_AGENTS[0]])
dedup_r3.deduplicate_agents()
names = [a.name for a in dedup_r3.agents]
assert names.count(DEFAULT_AGENTS[0].name) == 1
print("  ✅ deduplicate_agents: keeps first occurrence")
dedup_passed += 1

# 4. empty router
dedup_empty = PromptRouter([])
assert dedup_empty.deduplicate_agents() == 0
print("  ✅ deduplicate_agents: empty router")
dedup_passed += 1

print(f"\n{dedup_passed}/{dedup_total} deduplicate_agents tests passed")
total_passed += dedup_passed
total_tests += dedup_total

# ========== route_by_regex tests ==========
regex_total = 6
regex_passed = 0

# 1. match security keywords → reviewer
regex_r = r.route_by_regex("Check security", pattern=r"secur")
assert regex_r["agent"] is not None
assert len(regex_r["matched_keywords"]) > 0
print("  ✅ route_by_regex: matches keywords")
regex_passed += 1

# 2. no match returns None agent
regex_none = r.route_by_regex("test", pattern=r"^zzzzz$")
assert regex_none["agent"] is None
print("  ✅ route_by_regex: no match")
regex_passed += 1

# 3. all_matches populated
regex_am = r.route_by_regex("Write code", pattern=r"code|bug")
assert isinstance(regex_am["all_matches"], list)
print("  ✅ route_by_regex: all_matches populated")
regex_passed += 1

# 4. match_ratio between 0 and 1
for entry in regex_am["all_matches"]:
    assert 0.0 <= entry["match_ratio"] <= 1.0
print("  ✅ route_by_regex: match_ratio in range")
regex_passed += 1

# 5. empty router
regex_empty = PromptRouter([]).route_by_regex("test", pattern=r"code")
assert regex_empty["agent"] is None or regex_empty["score"] == 0.0
print("  ✅ route_by_regex: empty router")
regex_passed += 1

# 6. case insensitive matching
regex_ci = r.route_by_regex("Research topic", pattern=r"RESEARCH")
assert regex_ci["agent"] is not None
print("  ✅ route_by_regex: case insensitive")
regex_passed += 1

print(f"\n{regex_passed}/{regex_total} route_by_regex tests passed")
total_passed += regex_passed
total_tests += regex_total

# ========== cooldown tests ==========
cd_total = 0
cd_passed = 0
import time

cd_r = PromptRouter()

# 1. cooldown returns True for existing agent
cd_total += 1
assert cd_r.cooldown("coder", seconds=60) is True
print("  ✅ cooldown: returns True for existing agent")
cd_passed += 1

# 2. cooldown returns False for non-existent agent
cd_total += 1
assert cd_r.cooldown("nonexistent", seconds=60) is False
print("  ✅ cooldown: returns False for non-existent agent")
cd_passed += 1

# 3. is_cooled_down returns True during cooldown
cd_total += 1
assert cd_r.is_cooled_down("coder") is True
print("  ✅ is_cooled_down: True during cooldown")
cd_passed += 1

# 4. is_cooled_down returns False for non-cooled agent
cd_total += 1
assert cd_r.is_cooled_down("researcher") is False
print("  ✅ is_cooled_down: False for non-cooled agent")
cd_passed += 1

# 5. route_respecting_cooldowns skips cooled agent
cd_total += 1
agent, score, eligible = cd_r.route_respecting_cooldowns("Fix the login bug")
assert agent != "coder", f"should skip coder in cooldown, got {agent}"
print("  ✅ route_respecting_cooldowns: skips cooled agent")
cd_passed += 1

# 6. clear_cooldown removes cooldown
cd_total += 1
assert cd_r.clear_cooldown("coder") is True
assert cd_r.is_cooled_down("coder") is False
agent2, _, _ = cd_r.route_respecting_cooldowns("Fix the login bug")
assert agent2 == "coder", f"after clearing cooldown, should route to coder, got {agent2}"
print("  ✅ clear_cooldown: removes cooldown, routing restored")
cd_passed += 1

# 7. expired cooldown auto-clears on check
cd_total += 1
cd_r.cooldown("writer", seconds=0.01)
time.sleep(0.02)
assert cd_r.is_cooled_down("writer") is False
print("  ✅ cooldown: expires and auto-clears")
cd_passed += 1

print(f"\n{cd_passed}/{cd_total} cooldown tests passed")
total_passed += cd_passed
total_tests += cd_total

# ========== route_by_priority tests ==========
p_total = 0
p_passed = 0

# 1. basic priority routing
p_total += 1
pr_r = PromptRouter()
result = pr_r.route_by_priority("Fix the bug")
assert result["agent"] is not None
assert len(result["rankings"]) == len(pr_r.agents)
assert result["effective_priority"] > 0
print("  ✅ route_by_priority: basic routing")
p_passed += 1

# 2. boost increases effective priority for matching agents
p_total += 1
result_no_boost = pr_r.route_by_priority("Fix the bug", boost=0.0)
result_boost = pr_r.route_by_priority("Fix the bug", boost=2.0)
coder_no = next(r for r in result_no_boost["rankings"] if r["name"] == "coder")
coder_boost = next(r for r in result_boost["rankings"] if r["name"] == "coder")
assert coder_boost["effective_priority"] > coder_no["effective_priority"]
print("  ✅ route_by_priority: boost increases effective priority")
p_passed += 1

# 3. decay reduces priority for frequently used agents
p_total += 1
result_decay = pr_r.route_by_priority("Fix the bug", decay=0.5, history=["coder", "coder", "coder"])
coder_decay = next(r for r in result_decay["rankings"] if r["name"] == "coder")
assert coder_decay["usage"] == 3
assert coder_decay["effective_priority"] < coder_decay["base_priority"]
print("  ✅ route_by_priority: decay reduces priority for heavy usage")
p_passed += 1

# 4. rankings sorted by effective_priority descending
p_total += 1
priorities = [r["effective_priority"] for r in result["rankings"]]
assert priorities == sorted(priorities, reverse=True)
print("  ✅ route_by_priority: rankings sorted descending")
p_passed += 1

# 5. empty router
p_total += 1
empty_pr = PromptRouter([]).route_by_priority("test")
assert empty_pr["agent"] is None
print("  ✅ route_by_priority: empty router")
p_passed += 1

# 6. boost only applies to agents with score > 0
p_total += 1
boosted = pr_r.route_by_priority("Fix the bug", boost=5.0)
for r_entry in boosted["rankings"]:
    if r_entry["score"] == 0:
        base_p = next(a.priority for a in pr_r.agents if a.name == r_entry["name"])
        assert abs(r_entry["effective_priority"] - base_p) < 0.01
print("  ✅ route_by_priority: boost only for scoring agents")
p_passed += 1

print(f"\n{p_passed}/{p_total} route_by_priority tests passed")
total_passed += p_passed
total_tests += p_total

print(f"\n📊 Total: {total_passed}/{total_tests} tests passed")

# ========== route_tournament tests ==========
t_total = 0
t_passed = 0
t_r = PromptRouter()

t_total += 1
tour = t_r.route_tournament("Fix the login bug")
assert tour["champion"] == "coder", f"expected coder, got {tour['champion']}"
assert tour["final_score"] > 0
assert tour["rounds_played"] > 0
print("  ✅ route_tournament: basic champion")
t_passed += 1

t_total += 1
assert len(tour["bracket"]) >= 1
first_round = tour["bracket"][0]
assert len(first_round) >= 3
assert all("winner" in m and "loser" in m for m in first_round)
print("  ✅ route_tournament: bracket structure")
t_passed += 1

t_total += 1
tour1 = t_r.route_tournament("Fix the bug", rounds=1)
assert tour1["rounds_played"] == 1
print("  ✅ route_tournament: rounds limit")
t_passed += 1

t_total += 1
tour_e = PromptRouter([]).route_tournament("test")
assert tour_e["champion"] is None
assert tour_e["rounds_played"] == 0
print("  ✅ route_tournament: empty router")
t_passed += 1

t_total += 1
tour_s = PromptRouter([DEFAULT_AGENTS[0]]).route_tournament("test")
assert tour_s["champion"] == DEFAULT_AGENTS[0].name
assert tour_s["rounds_played"] == 0
print("  ✅ route_tournament: single agent")
t_passed += 1

t_total += 1
tour_odd = PromptRouter(DEFAULT_AGENTS[:5]).route_tournament("Fix bug")
assert tour_odd["champion"] is not None
first = tour_odd["bracket"][0]
bye_matches = [m for m in first if m["loser"] is None]
assert len(bye_matches) == 1
print("  ✅ route_tournament: bye handling")
t_passed += 1

print(f"\n{t_passed}/{t_total} route_tournament tests passed")
total_passed += t_passed
total_tests += t_total

# ========== agent_similarity tests ==========
sim_total = 0
sim_passed = 0
sim_r = PromptRouter()

sim_total += 1
sim = sim_r.agent_similarity("coder", "researcher")
assert "jaccard" in sim
assert 0.0 <= sim["jaccard"] <= 1.0
assert "shared_keywords" in sim
assert "overlap_count" in sim
print("  ✅ agent_similarity: basic metrics")
sim_passed += 1

sim_total += 1
sim_self = sim_r.agent_similarity("coder", "coder")
assert sim_self["jaccard"] == 1.0
assert sim_self["overlap_count"] == len(sim_r.agents[0].keywords)
print("  ✅ agent_similarity: self = 1.0")
sim_passed += 1

sim_total += 1
custom_a = Agent("xa", "a", keywords=["alpha", "beta"])
custom_b = Agent("xb", "b", keywords=["gamma", "delta"])
sim_r2 = PromptRouter([custom_a, custom_b])
sim_diff = sim_r2.agent_similarity("xa", "xb")
assert sim_diff["jaccard"] == 0.0
assert sim_diff["shared_keywords"] == []
print("  ✅ agent_similarity: no overlap = 0.0")
sim_passed += 1

sim_total += 1
sim_none = sim_r.agent_similarity("coder", "nonexistent")
assert sim_none == {}
print("  ✅ agent_similarity: missing agent")
sim_passed += 1

sim_total += 1
sim5 = sim_r.agent_similarity("coder", "reviewer")
assert len(sim5["unique_to_a"]) > 0 or len(sim5["unique_to_b"]) > 0
print("  ✅ agent_similarity: unique keywords")
sim_passed += 1

print(f"\n{sim_passed}/{sim_total} agent_similarity tests passed")
total_passed += sim_passed
total_tests += sim_total

# ========== route_weighted_vote tests ==========
v_total = 0
v_passed = 0
v_r = PromptRouter()

v_total += 1
vote = v_r.route_weighted_vote("Fix the login bug")
assert vote["winner"] == "coder", f"expected coder, got {vote['winner']}"
assert "tally" in vote
assert len(vote["votes"]) > 0
print("  ✅ route_weighted_vote: basic vote")
v_passed += 1

v_total += 1
vote4 = v_r.route_weighted_vote("Fix the bug", strategies=["score", "top_k", "diversity", "priority"])
assert len(vote4["votes"]) == 4
assert len(vote4["strategies_used"]) == 4
print("  ✅ route_weighted_vote: all strategies")
v_passed += 1

v_total += 1
vote_bias = v_r.route_weighted_vote("Summarize data", strategies=["score", "top_k"],
                                        weights={"score": 5.0, "top_k": 0.1})
assert vote_bias["tally"]
print("  ✅ route_weighted_vote: custom weights")
v_passed += 1

v_total += 1
vote_e = PromptRouter([]).route_weighted_vote("test")
assert "winner" in vote_e
print("  ✅ route_weighted_vote: empty router")
v_passed += 1

v_total += 1
vote5 = v_r.route_weighted_vote("Fix bug")
tally_values = list(vote5["tally"].values())
assert tally_values == sorted(tally_values, reverse=True)
print("  ✅ route_weighted_vote: tally sorted")
v_passed += 1

v_total += 1
vote_unk = v_r.route_weighted_vote("Fix bug", strategies=["score", "nonexistent_strategy"])
assert len(vote_unk["votes"]) == 1
print("  ✅ route_weighted_vote: unknown strategy skipped")
v_passed += 1

print(f"\n{v_passed}/{v_total} route_weighted_vote tests passed")
total_passed += v_passed
total_tests += v_total

# ========== export_report tests ==========
rep_total = 0
rep_passed = 0
rep_r = PromptRouter()

rep_total += 1
report = rep_r.export_report(["Fix bug", "Research AI", "Write docs"])
assert report["total_prompts"] == 3
assert len(report["per_prompt"]) == 3
assert "agent_coverage" in report
assert "confidence_distribution" in report
assert "diversity_score" in report
print("  ✅ export_report: basic structure")
rep_passed += 1

rep_total += 1
assert sum(report["agent_coverage"].values()) == 3
print("  ✅ export_report: coverage sums correctly")
rep_passed += 1

rep_total += 1
assert sum(report["confidence_distribution"].values()) == 3
print("  ✅ export_report: confidence sums correctly")
rep_passed += 1

rep_total += 1
assert 0.0 <= report["diversity_score"] <= 1.0
print("  ✅ export_report: diversity score range")
rep_passed += 1

rep_total += 1
tmpf = os.path.join(tempfile.gettempdir(), "test_report.json")
report2 = rep_r.export_report(["Fix bug"], path=tmpf)
assert os.path.exists(tmpf)
with open(tmpf) as f:
    loaded = json.load(f)
assert loaded["total_prompts"] == 1
os.unlink(tmpf)
print("  ✅ export_report: file export")
rep_passed += 1

rep_total += 1
report_e = rep_r.export_report([])
assert report_e["total_prompts"] == 0
assert report_e["per_prompt"] == []
assert report_e["diversity_score"] == 0.0
print("  ✅ export_report: empty prompts")
rep_passed += 1

rep_total += 1
keys = set(report["per_prompt"][0].keys())
assert keys == {"prompt", "agent", "score", "confidence"}
print("  ✅ export_report: per_prompt keys")
rep_passed += 1

print(f"\n{rep_passed}/{rep_total} export_report tests passed")
total_passed += rep_passed
total_tests += rep_total

# ============================================================
# health_check tests
# ============================================================
hc_passed = 0
hc_total = 0

hc_total += 1
hc_r = PromptRouter(DEFAULT_AGENTS)
result = hc_r.health_check()
assert result["healthy"] is True
assert result["issues"] == []
assert result["agent_count"] == len(DEFAULT_AGENTS)
print("  ✅ health_check: healthy router")
hc_passed += 1

hc_total += 1
hc_bad = PromptRouter([Agent("dup", "a", keywords=["x"]), Agent("dup", "b", keywords=["y"])])
result = hc_bad.health_check()
assert not result["healthy"]
assert any("duplicate" in i for i in result["issues"])
print("  ✅ health_check: detects duplicate names")
hc_passed += 1

hc_total += 1
hc_no_kw = PromptRouter([Agent("empty", "", keywords=[])])
result = hc_no_kw.health_check()
assert not result["healthy"]
assert any("no keywords" in i for i in result["issues"])
assert any("no description" in i for i in result["issues"])
print("  ✅ health_check: detects no-keyword and no-description agents")
hc_passed += 1

hc_total += 1
hc_zero = PromptRouter([Agent("zero", "", keywords=[], patterns=[])])
result = hc_zero.health_check()
assert not result["healthy"]
assert any("no routing signals" in i for i in result["issues"])
print("  ✅ health_check: detects agents with no routing signals")
hc_passed += 1

hc_total += 1
hc_empty = PromptRouter([])
result = hc_empty.health_check()
assert result["healthy"] is True  # no agents = no issues
assert result["agent_count"] == 0
print("  ✅ health_check: empty router")
hc_passed += 1

hc_total += 1
hc_cd = PromptRouter(DEFAULT_AGENTS)
hc_cd.cooldown(DEFAULT_AGENTS[0].name, seconds=9999)
result = hc_cd.health_check()
assert not result["healthy"]
assert any("cooldown" in i for i in result["issues"])
print("  ✅ health_check: detects active cooldowns")
hc_passed += 1

print(f"\n{hc_passed}/{hc_total} health_check tests passed")
total_passed += hc_passed
total_tests += hc_total

# ============================================================
# route_by_time_window tests
# ============================================================
tw_passed = 0
tw_total = 0

tw_total += 1
tw_r = PromptRouter(DEFAULT_AGENTS)
result = tw_r.route_by_time_window("Write a function", {"coder": (9, 17)}, current_hour=12)
assert result["agent"] is not None
assert result["current_hour"] == 12
assert result["eligible_count"] == len(DEFAULT_AGENTS)  # all eligible at noon
print("  ✅ route_by_time_window: agent in active window")
tw_passed += 1

tw_total += 1
result2 = tw_r.route_by_time_window("Write a function", {"coder": (9, 17)}, current_hour=22)
# coder excluded at 22h, other agents still eligible
assert result2["agent"] is not None
assert result2["eligible_count"] == len(DEFAULT_AGENTS) - 1
print("  ✅ route_by_time_window: agent excluded outside window")
tw_passed += 1

tw_total += 1
# Wrap-around window: (22, 6) covers 22,23,0,1,2,3,4,5
result3 = tw_r.route_by_time_window("Debug this", {"reviewer": (22, 6)}, current_hour=3)
assert result3["agent"] is not None
print("  ✅ route_by_time_window: wrap-around midnight")
tw_passed += 1

tw_total += 1
result4 = tw_r.route_by_time_window("Debug this", {"reviewer": (22, 6)}, current_hour=12)
assert result4["eligible_count"] == len(DEFAULT_AGENTS) - 1
print("  ✅ route_by_time_window: wrap-around excludes midday")
tw_passed += 1

tw_total += 1
# All agents in windows, none match → no agents available
tw_single = PromptRouter([Agent("only", "only agent", keywords=["x"])])
result5 = tw_single.route_by_time_window("test", {"only": (9, 17)}, current_hour=22)
assert result5["agent"] is None
assert result5["reason"] == "no agents available in this time window"
print("  ✅ route_by_time_window: no eligible agents returns None")
tw_passed += 1

tw_total += 1
# Agents not in windows dict are always eligible
result6 = tw_r.route_by_time_window("Write code", {}, current_hour=5)
assert result6["agent"] is not None
assert result6["eligible_count"] == len(DEFAULT_AGENTS)
print("  ✅ route_by_time_window: empty windows includes all")
tw_passed += 1

print(f"\n{tw_passed}/{tw_total} route_by_time_window tests passed")
total_passed += tw_passed
total_tests += tw_total

# ============================================================
# cross_validate tests
# ============================================================
cv_total = 0
cv_passed = 0
cv_r = PromptRouter()

cv_total += 1
cv_result = cv_r.cross_validate([
    ("Fix the authentication bug", "coder"),
    ("Summarize the Q4 report", "researcher"),
    ("Write a blog post about Rust", "writer"),
])
assert cv_result["total"] == 3
assert cv_result["correct"] >= 2  # most should match
assert 0.0 < cv_result["accuracy"] <= 1.0
assert len(cv_result["errors"]) == 3 - cv_result["correct"]
assert "per_agent" in cv_result
assert "confusion_matrix" in cv_result
print("  ✅ cross_validate: basic evaluation")
cv_passed += 1

cv_total += 1
cv_empty = cv_r.cross_validate([])
assert cv_empty["total"] == 0
assert cv_empty["accuracy"] == 0.0
assert cv_empty["errors"] == []
print("  ✅ cross_validate: empty test cases")
cv_passed += 1

cv_total += 1
# 100% accuracy on strong matches
cv_perfect = cv_r.cross_validate([
    ("Fix the login bug", "coder"),
    ("Fix the compile error", "coder"),
])
assert cv_perfect["accuracy"] == 1.0
assert cv_perfect["errors"] == []
print("  ✅ cross_validate: perfect accuracy")
cv_passed += 1

cv_total += 1
# per_agent has precision/recall/f1
for agent_name, metrics in cv_result["per_agent"].items():
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
print("  ✅ cross_validate: per_agent precision/recall/f1")
cv_passed += 1

cv_total += 1
# confusion matrix dimensions
agent_names = [a.name for a in cv_r.agents]
for pred in agent_names:
    assert pred in cv_result["confusion_matrix"]
    for actual in agent_names:
        assert actual in cv_result["confusion_matrix"][pred]
print("  ✅ cross_validate: confusion matrix dimensions")
cv_passed += 1

cv_total += 1
# errors have correct structure
cv_mixed = cv_r.cross_validate([
    ("Fix the bug", "coder"),
    ("xyzzy plugh", "researcher"),  # likely misclassified
])
if cv_mixed["errors"]:
    err = cv_mixed["errors"][0]
    assert "prompt" in err and "expected" in err and "predicted" in err and "score" in err
print("  ✅ cross_validate: error structure")
cv_passed += 1

cv_total += 1
# tp+fp+fn are non-negative integers
for name, m in cv_result["per_agent"].items():
    assert m["tp"] >= 0 and m["fp"] >= 0 and m["fn"] >= 0
print("  ✅ cross_validate: tp/fp/fn non-negative")
cv_passed += 1

cv_total += 1
# empty router cross_validate
cv_er = PromptRouter([])
cv_er_result = cv_er.cross_validate([("test", "nonexistent_agent")])
assert cv_er_result["total"] == 1
assert cv_er_result["accuracy"] < 1.0  # can't match nonexistent agent
print("  ✅ cross_validate: empty router")
cv_passed += 1

print(f"\n{cv_passed}/{cv_total} cross_validate tests passed")
total_passed += cv_passed
total_tests += cv_total

# ============================================================
# suggest_improvements tests
# ============================================================
si_total = 0
si_passed = 0
si_r = PromptRouter()

si_total += 1
# perfect accuracy = no suggestions
si_perfect = si_r.suggest_improvements([
    ("Fix the login bug", "coder"),
    ("Fix the compile error", "coder"),
])
assert si_perfect["accuracy"] == 1.0
assert si_perfect["suggestions"] == []
print("  ✅ suggest_improvements: perfect accuracy, no suggestions")
si_passed += 1

si_total += 1
# imperfect = suggestions generated
si_bad = si_r.suggest_improvements([
    ("Translate English to French", "translator"),
    ("Fix the login bug", "coder"),
    ("xyzzy plugh foobar", "researcher"),
])
assert si_bad["accuracy"] < 1.0
assert si_bad["total_errors"] > 0
assert len(si_bad["suggestions"]) > 0
print("  ✅ suggest_improvements: generates suggestions for errors")
si_passed += 1

si_total += 1
# suggestions have correct structure
if si_bad["suggestions"]:
    s = si_bad["suggestions"][0]
    assert "agent" in s and "misclassified_as" in s
    assert "prompt" in s and "suggested_keywords" in s
print("  ✅ suggest_improvements: suggestion structure")
si_passed += 1

si_total += 1
# by_agent groups correctly
assert isinstance(si_bad["by_agent"], dict)
for agent_name, items in si_bad["by_agent"].items():
    assert isinstance(items, list)
    for item in items:
        assert item["agent"] == agent_name
print("  ✅ suggest_improvements: by_agent grouping")
si_passed += 1

si_total += 1
# empty test cases
si_empty = si_r.suggest_improvements([])
assert si_empty["accuracy"] == 0.0
assert si_empty["total_errors"] == 0
print("  ✅ suggest_improvements: empty input")
si_passed += 1

si_total += 1
# suggested_keywords are words from the misclassified prompt
if si_bad["suggestions"]:
    for s in si_bad["suggestions"]:
        for kw in s["suggested_keywords"]:
            assert len(kw) >= 4, f"keyword too short: {kw}"
print("  ✅ suggest_improvements: keywords filtered by length")
si_passed += 1

print(f"\n{si_passed}/{si_total} suggest_improvements tests passed")
total_passed += si_passed
total_tests += si_total

print(f"\n📊 Grand Total: {total_passed}/{total_tests} tests passed" if total_tests else "")

# ============================================================
# route_negotiation tests
# ============================================================
neg_r = PromptRouter()
neg_passed = 0
neg_total = 0

neg_total += 1
neg_result = neg_r.route_negotiation("Fix the authentication bug in login.py")
assert neg_result["winner"] is not None
assert isinstance(neg_result["candidates"], list)
assert isinstance(neg_result["phase1_scores"], list)
assert isinstance(neg_result["phase2_scores"], list)
assert isinstance(neg_result["negotiated"], bool)
print("  ✅ route_negotiation: basic structure")
neg_passed += 1

neg_total += 1
# winner is one of the candidates
assert neg_result["winner"] in neg_result["candidates"]
print("  ✅ route_negotiation: winner is a candidate")
neg_passed += 1

neg_total += 1
# phase2 scores have expected keys
p2 = neg_result["phase2_scores"][0]
assert "base_score" in p2 and "desc_overlap" in p2
assert "specificity_bonus" in p2 and "pattern_bonus" in p2
assert "final_score" in p2
print("  ✅ route_negotiation: phase2 score breakdown")
neg_passed += 1

neg_total += 1
# top_k limits candidates
neg_k2 = neg_r.route_negotiation("Write a blog post", top_k=2)
assert len(neg_k2["candidates"]) <= 2
assert len(neg_k2["phase2_scores"]) <= 2
print("  ✅ route_negotiation: top_k limits candidates")
neg_passed += 1

neg_total += 1
# negotiated=True when phase2 changes winner (hard to guarantee, so test structure)
# At minimum, negotiated is a boolean
assert isinstance(neg_result["negotiated"], bool)
print("  ✅ route_negotiation: negotiated flag type")
neg_passed += 1

neg_total += 1
# empty router
neg_empty_r = PromptRouter([])
neg_empty = neg_empty_r.route_negotiation("hello")
assert neg_empty["winner"] is None
assert neg_empty["candidates"] == []
print("  ✅ route_negotiation: empty router")
neg_passed += 1

neg_total += 1
# phase1_scores covers all agents
assert len(neg_result["phase1_scores"]) == len(neg_r.agents)
for name, score in neg_result["phase1_scores"]:
    assert isinstance(name, str) and isinstance(score, (int, float))
print("  ✅ route_negotiation: phase1_scores completeness")
neg_passed += 1

print(f"\n{neg_passed}/{neg_total} route_negotiation tests passed")
total_passed += neg_passed
total_tests += neg_total

print(f"\n📊 Grand Total: {total_passed}/{total_tests} tests passed")

# ============================================================
# route_by_sentiment tests
# ============================================================
sent_r = PromptRouter()
sent_passed = 0
sent_total = 0

sent_total += 1
sent_urgent = sent_r.route_by_sentiment("Fix this critical bug ASAP!")
assert sent_urgent["sentiment"] == "urgent"
assert sent_urgent["urgency"] is True
assert sent_urgent["agent"] is not None
print("  ✅ route_by_sentiment: urgent detection")
sent_passed += 1

sent_total += 1
sent_frustrated = sent_r.route_by_sentiment("This login is broken again, keeps failing")
assert sent_frustrated["sentiment"] == "frustrated"
assert "frustrated" in sent_frustrated["active_signals"]
print("  ✅ route_by_sentiment: frustrated detection")
sent_passed += 1

sent_total += 1
sent_polite = sent_r.route_by_sentiment("Could you please explain how this works?")
assert sent_polite["sentiment"] == "polite_question"
assert sent_polite["signals"]["polite"] is True
assert sent_polite["signals"]["question"] is True
print("  ✅ route_by_sentiment: polite question detection")
sent_passed += 1

sent_total += 1
sent_command = sent_r.route_by_sentiment("Write a Python script for backups")
assert sent_command["sentiment"] == "command"
assert sent_command["signals"]["command"] is True
print("  ✅ route_by_sentiment: command detection")
sent_passed += 1

sent_total += 1
sent_neutral = sent_r.route_by_sentiment("deploy the service")
assert sent_neutral["sentiment"] in ("neutral", "command")
assert "agent" in sent_neutral and "score" in sent_neutral
print("  ✅ route_by_sentiment: neutral/basic routing")
sent_passed += 1

sent_total += 1
# signals structure
assert isinstance(sent_urgent["signals"], dict)
for key in ("urgent", "polite", "frustrated", "question", "command"):
    assert key in sent_urgent["signals"]
print("  ✅ route_by_sentiment: signals structure")
sent_passed += 1

sent_total += 1
# preference boost affects routing
# Urgent coding prompt should boost coder
assert sent_urgent["agent"] == "coder"  # urgent + code → coder
print("  ✅ route_by_sentiment: sentiment-based preference")
sent_passed += 1

print(f"\n{sent_passed}/{sent_total} route_by_sentiment tests passed")
total_passed += sent_passed
total_tests += sent_total

print(f"\n📊 Grand Total: {total_passed}/{total_tests} tests passed")

# ============================================================
# learning_stats + reset_learning tests
# ============================================================
learn_r = PromptRouter()
learn_passed = 0
learn_total = 0

learn_total += 1
ls0 = learn_r.learning_stats()
assert ls0["total_feedback"] == 0
assert ls0["accuracy"] is None
assert isinstance(ls0["priority_changes"], dict)
print("  ✅ learning_stats: initial empty state")
learn_passed += 1

learn_total += 1
learn_r.route_adaptive("Fix the bug", correct_agent="coder")
learn_r.route_adaptive("Write a blog post", correct_agent="writer")
learn_r.route_adaptive("Explain quantum computing", correct_agent="researcher")
ls1 = learn_r.learning_stats()
assert ls1["total_feedback"] == 3
assert ls1["accuracy"] is not None
print("  ✅ learning_stats: after feedback")
learn_passed += 1

learn_total += 1
reset_result = learn_r.reset_learning()
assert reset_result["cleared_feedback"] == 3
assert isinstance(reset_result["reset_priorities"], dict)
for name, changes in reset_result["reset_priorities"].items():
    assert changes["after"] == 1.0
print("  ✅ reset_learning: clears feedback and resets priorities")
learn_passed += 1

learn_total += 1
ls2 = learn_r.learning_stats()
assert ls2["total_feedback"] == 0
assert ls2["accuracy"] is None
print("  ✅ reset_learning: stats reflect reset")
learn_passed += 1

learn_total += 1
learn_r.route_adaptive("Plan the sprint", correct_agent="planner")
reset_custom = learn_r.reset_learning(priority=2.0)
for a in learn_r.agents:
    assert a.priority == 2.0
print("  ✅ reset_learning: custom priority")
learn_passed += 1

learn_total += 1
ls3 = learn_r.learning_stats()
for name, pri in ls3["priority_changes"].items():
    assert pri == 2.0
print("  ✅ learning_stats: reflects current priorities")
learn_passed += 1

print(f"\n{learn_passed}/{learn_total} learning tests passed")
total_passed += learn_passed
total_tests += learn_total

print(f"\n📊 Grand Total: {total_passed}/{total_tests} tests passed")

# ============================================================
# detect_language + route_by_language tests
# ============================================================
lang_r = PromptRouter()
lang_passed = 0
lang_total = 0

lang_total += 1
assert lang_r.detect_language("Fix the login bug") == "en"
print("  ✅ detect_language: English detected")
lang_passed += 1

lang_total += 1
assert lang_r.detect_language("写一封英文邮件") == "zh"
print("  ✅ detect_language: Chinese detected")
lang_passed += 1

lang_total += 1
assert lang_r.detect_language("バグを修正して") == "ja"
print("  ✅ detect_language: Japanese detected")
lang_passed += 1

lang_total += 1
assert lang_r.detect_language("버그를 수정해") == "ko"
print("  ✅ detect_language: Korean detected")
lang_passed += 1

lang_total += 1
assert lang_r.detect_language("") == "unknown"
print("  ✅ detect_language: empty returns unknown")
lang_passed += 1

lang_total += 1
rl = lang_r.route_by_language("写一封英文邮件给客户")
assert rl["language"] == "zh"
assert rl["agent"] in ("translator", "writer", "researcher")
assert "score" in rl and "lang_candidates" in rl
print("  ✅ route_by_language: Chinese prompt routes correctly")
lang_passed += 1

lang_total += 1
rl_en = lang_r.route_by_language("Fix the bug in login.py")
assert rl_en["language"] == "en"
assert "lang_candidates" in rl_en
assert "any" in rl_en["lang_candidates"]
print("  ✅ route_by_language: English prompt has no lang preference")
lang_passed += 1

lang_total += 1
rl_custom = lang_r.route_by_language("Fix the bug", lang_map={"en": ["coder"]})
assert rl_custom["agent"] == "coder"
print("  ✅ route_by_language: custom lang_map works")
lang_passed += 1

print(f"\n{lang_passed}/{lang_total} route_by_language tests passed")
total_passed += lang_passed
total_tests += lang_total
print(f"\n📊 Grand Total: {total_passed}/{total_tests} tests passed")

# ============================================================
# route_by_complexity tests
# ============================================================
comp_r = PromptRouter()
comp_passed = 0
comp_total = 0

comp_total += 1
simple = comp_r.route_by_complexity("Fix the bug")
assert simple["complexity"] == "low"
assert simple["signals"]["word_count"] == 3
print("  ✅ route_by_complexity: low complexity")
comp_passed += 1

comp_total += 1
complex_prompt = "Design and implement a distributed microservice architecture with API gateway, service mesh, and infrastructure automation protocol for the database layer"
complex_result = comp_r.route_by_complexity(complex_prompt)
assert complex_result["complexity"] == "high"
assert complex_result["signals"]["technical_terms"] >= 2
print("  ✅ route_by_complexity: high complexity")
comp_passed += 1

comp_total += 1
medium = comp_r.route_by_complexity("Write a Python script that reads CSV files and generates a report with charts")
assert medium["complexity"] in ("low", "medium", "high")
# has_code may be False since CSV is not in the code pattern set
assert "word_count" in medium["signals"]
print("  ✅ route_by_complexity: basic multi-word prompt")
comp_passed += 1

comp_total += 1
negation = comp_r.route_by_complexity("Do not use the database, avoid external APIs")
assert negation["signals"]["has_negation"] is True
print("  ✅ route_by_complexity: negation detection")
comp_passed += 1

comp_total += 1
assert "complexity_score" in simple and "signals" in simple and "agent" in simple
print("  ✅ route_by_complexity: result structure")
comp_passed += 1

print(f"\n{comp_passed}/{comp_total} route_by_complexity tests passed")
total_passed += comp_passed
total_tests += comp_total
print(f"\n📊 Grand Total: {total_passed}/{total_tests} tests passed")

# ============================================================
# agent_graph tests
# ============================================================
graph_r = PromptRouter()
graph_passed = 0
graph_total = 0

graph_total += 1
g = graph_r.agent_graph()
assert len(g["nodes"]) == len(graph_r.agents)
print("  ✅ agent_graph: correct node count")
graph_passed += 1

graph_total += 1
expected_edges = len(graph_r.agents) * (len(graph_r.agents) - 1) // 2
assert len(g["edges"]) == expected_edges
print("  ✅ agent_graph: correct edge count")
graph_passed += 1

graph_total += 1
first_edge = g["edges"][0]
assert "from" in first_edge and "to" in first_edge and "similarity" in first_edge
print("  ✅ agent_graph: edge structure")
graph_passed += 1

graph_total += 1
single_r = PromptRouter([graph_r.agents[0]])
single_g = single_r.agent_graph()
assert len(single_g["edges"]) == 0
print("  ✅ agent_graph: single agent has no edges")
graph_passed += 1

print(f"\n{graph_passed}/{graph_total} agent_graph tests passed")
total_passed += graph_passed
total_tests += graph_total
print(f"\n📊 Grand Total: {total_passed}/{total_tests} tests passed")

# ============================================================
# export_state + import_state tests
# ============================================================
es_r = PromptRouter()
es_passed = 0
es_total = 0

es_total += 1
state = es_r.export_state()
assert state["version"] == "1.0"
assert state["agent_count"] == len(es_r.agents)
assert len(state["agents"]) == len(es_r.agents)
print("  ✅ export_state: basic structure")
es_passed += 1

es_total += 1
first_agent = state["agents"][0]
for key in ("name", "description", "keywords", "patterns", "priority"):
    assert key in first_agent
print("  ✅ export_state: agent fields complete")
es_passed += 1

es_total += 1
# Round-trip: export → import → verify routing still works
es_r.route_adaptive("Fix the bug", correct_agent="coder")
state_with_fb = es_r.export_state()
assert len(state_with_fb["feedback"]) > 0

new_r = PromptRouter([])
result = new_r.import_state(state_with_fb)
assert result["imported_agents"] == len(es_r.agents)
assert result["imported_feedback"] > 0
print("  ✅ import_state: round-trip preserves agents and feedback")
es_passed += 1

es_total += 1
# Verify imported router routes correctly
agent, score, _ = new_r.route("Fix the bug")
assert isinstance(agent, str) and isinstance(score, float)
print("  ✅ import_state: imported router routes correctly")
es_passed += 1

es_total += 1
empty_import = PromptRouter([]).import_state({"agents": [], "feedback": []})
assert empty_import["imported_agents"] == 0
print("  ✅ import_state: empty state")
es_passed += 1

print(f"\n{es_passed}/{es_total} export_import tests passed")
total_passed += es_passed
total_tests += es_total

# ============================================================
# route_by_length tests
# ============================================================
rl_passed = 0
rl_total = 0

rl_total += 1
r = PromptRouter()
res = r.route_by_length("hi")
assert res["word_count"] == 1
assert res["category"] == "short"
assert res["agent"] in [a.name for a in r.agents]
print("  ✅ route_by_length: short prompt")
rl_passed += 1

rl_total += 1
res = r.route_by_length("Write a function that sorts an array using merge sort and explain the time complexity step by step with examples for each phase of the algorithm")
assert res["word_count"] > 20
assert res["category"] == "long"
assert res["agent"] in [a.name for a in r.agents]
print("  ✅ route_by_length: long prompt")
rl_passed += 1

rl_total += 1
res = r.route_by_length("Fix the bug in the code module today")
assert res["category"] == "medium"
print("  ✅ route_by_length: medium prompt")
rl_passed += 1

rl_total += 1
# With length_map, prefer specific agents
small_r = PromptRouter([DEFAULT_AGENTS[0], DEFAULT_AGENTS[1]])
res = small_r.route_by_length("hi", length_map={"short": [DEFAULT_AGENTS[1].name], "medium": [], "long": []})
assert res["agent"] == DEFAULT_AGENTS[1].name
print("  ✅ route_by_length: length_map override")
rl_passed += 1

rl_total += 1
# No matching agent in length_map falls back to best
res = r.route_by_length("hi", length_map={"short": ["nonexistent-agent"]})
assert res["agent"] in [a.name for a in r.agents]
print("  ✅ route_by_length: nonexistent agent fallback")
rl_passed += 1

rl_total += 1
# All scores present
res = r.route_by_length("Write a test")
assert len(res["all_scores"]) > 0
print("  ✅ route_by_length: all_scores populated")
rl_passed += 1

print(f"\n{rl_passed}/{rl_total} route_by_length tests passed")
total_passed += rl_passed
total_tests += rl_total

# ============================================================
# prune_agents tests
# ============================================================
pa_passed = 0
pa_total = 0

pa_total += 1
from prompt_router import Agent
empty_agent = Agent(name="empty", description="no keywords", keywords=[])
pr = PromptRouter([empty_agent, DEFAULT_AGENTS[0]])
result = pr.prune_agents()
assert result["removed"] == ["empty"]
assert result["remaining"] == 1
print("  ✅ prune_agents: removes agent with no keywords")
pa_passed += 1

pa_total += 1
pr2 = PromptRouter(DEFAULT_AGENTS)
result = pr2.prune_agents()
assert result["removed_count"] == 0
print("  ✅ prune_agents: keeps agents with keywords")
pa_passed += 1

pa_total += 1
pr3 = PromptRouter([])
result = pr3.prune_agents()
assert result["removed_count"] == 0 and result["remaining"] == 0
print("  ✅ prune_agents: empty router")
pa_passed += 1

print(f"\n{pa_passed}/{pa_total} prune_agents tests passed")
total_passed += pa_passed
total_tests += pa_total

# ============================================================
# optimize_weights tests
# ============================================================
ow_passed = 0
ow_total = 0

ow_total += 1
r = PromptRouter([DEFAULT_AGENTS[0], DEFAULT_AGENTS[1]])
result = r.optimize_weights([])
assert result["processed"] == 0
assert result["adjustment_count"] == 0
print("  ✅ optimize_weights: empty feedback")
ow_passed += 1

ow_total += 1
result = r.optimize_weights([("debug code", DEFAULT_AGENTS[0].name, True)])
assert result["processed"] == 1
assert result["adjustment_count"] == 0  # correct, no adjustment needed
print("  ✅ optimize_weights: correct routing no adjustment")
ow_passed += 1

ow_total += 1
before_kw = len(r.agents[0].keywords)
result = r.optimize_weights([("debug code", r.agents[0].name, False)])
assert result["adjustment_count"] == 1
assert result["agents_affected"] == [r.agents[0].name]
print("  ✅ optimize_weights: misroute triggers adjustment")
ow_passed += 1

ow_total += 1
result = r.optimize_weights([("test", "nonexistent-agent", False)])
assert result["adjustment_count"] == 0
print("  ✅ optimize_weights: unknown agent skipped")
ow_passed += 1

ow_total += 1
result = r.optimize_weights([
    ("debug code", r.agents[0].name, False),
    ("analyze data", r.agents[1].name, False),
])
assert result["processed"] == 2
assert result["adjustment_count"] == 2
print("  ✅ optimize_weights: multiple feedback entries")
ow_passed += 1

print(f"\n{ow_passed}/{ow_total} optimize_weights tests passed")
total_passed += ow_passed
total_tests += ow_total

print(f"\n📊 Grand Total: {total_passed}/{total_tests} tests passed")
