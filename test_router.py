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

total_passed = passed + explain_tests_passed + conf_tests_passed + batch_passed
total_tests = len(tests) + explain_tests_total + conf_tests_total + batch_total
print(f"\n📊 Total: {total_passed}/{total_tests} tests passed")
