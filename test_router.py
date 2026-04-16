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
total_passed = passed + explain_tests_passed
total_tests = len(tests) + explain_tests_total
print(f"\n📊 Total: {total_passed}/{total_tests} tests passed")
