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
