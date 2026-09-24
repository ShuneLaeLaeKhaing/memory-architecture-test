# Memory Architecture — Findings & Current Status

**Audience:** Team lead  
**Scope:** `memory-architecture-test` playground (Mem0 semantic + episodes + retrieval)  
**Date:** Sep 2026  

---

## 1. What this system is

Enterprise-style agent memory for personal facts and past conversations:

```
Chat → Events → (later) Consolidate → Episodes → Promote → Mem0 facts
                ↑
         Answer path: retrieve facts/episodes + thread history → reply
```

| Layer | Role | Storage |
|--------|------|---------|
| **Events** | Raw chat log | Local JSONL |
| **Thread history** | Last N turns (working memory) | Events by thread |
| **Episodes** | Summarized conversation chunks | Disk JSON + Chroma |
| **Semantic facts** | Durable prefs / identity | **Mem0 Cloud** (not Postgres) |
| **Procedural** | Approved skills / SOPs | Markdown + Chroma |
| **Knowledge base** | Handbook / company docs | Local KB + Chroma |

**Important:** User semantic facts are **user XOR agent** (no shared org identity in Mem0). Company rules stay in KB / procedures. Sensitivity tags (`public` vs `agent_only`) limit cross-agent leaks on private user facts.

---

## 2. How a turn works (answer path)

1. Log user/agent messages to the event store.  
2. **Route** query → semantic / episodic / procedural budgets (LLM, with keyword fallback).  
3. **Retrieve** user + this agent’s facts; apply authority (user > agent; same `fact_key` → newer).  
4. **Conflict check** (prefs like theme / notify): compare thread timestamps vs fact `as_of`.  
5. LLM answers with authoritative context + recent thread.

**Conflict policy (same user):**

| Situation | Winner |
|-----------|--------|
| New chat statement vs older saved fact | **Current chat** (until consolidate) |
| Old chat / stale episode vs newer saved fact | **Semantic fact** |
| Two facts, same key | **Newer** (supersede + recall safety net) |

---

## 3. How learning works (consolidate → promote)

**Triggers (no live per-message topic LLM — cost):**

- Manual `/consolidate`  
- End phrases (`bye` / `thanks`)  
- Idle ~5 minutes  
- N-turn event count (e.g. HR 20 / Onboarding 15)  
- Thread switch / exit  

**Inside consolidate:**

1. Segment events (size + optional topic split **only here**).  
2. Synthesize episode (LLM).  
3. Extract fact candidates with `fact_key` + value (LLM).  
4. Policy: promote / merge / supersede / discard.  
5. Write to Mem0.

Topic change is **not** run on every user message; N-turn thread history covers short continuity.

---

## 4. Key findings from testing

### 4.1 Deterministic vs heuristic vs LLM

We do **not** need an LLM on every step. Rough split:

- **Deterministic:** event log, Mem0 R/W, budgets, synonym map, supersede, authority ranking.  
- **Heuristic:** end/idle/N-turn triggers, transient/generic gates, conflict keyword topics, stem key match, **evidence grounding**.  
- **LLM:** route, episode synthesize, extract candidates, key-match fallback, final reply.

Burmese support does **not** require a parallel “MY-only” stack; mixed MY+EN already flows through extract/answer. Gaps are usually **grounding** or **key identity**, not “missing a Burmese LLM everywhere.”

### 4.2 Failure categories (how we debug)

When something looks wrong, label it first:

| Category | Meaning | Example |
|----------|---------|---------|
| **Extraction** | Wrong fact pulled from episode | Invented salary when user only said theme |
| **Key-identity** | Same concept, different `fact_key` → no supersede | `ui_theme` vs `preferred_ui_theme` |
| **Scope** | Wrong owner | User pref written as org/agent |
| **Evidence / promotion** | Value not in episode text but still written | Fake token promoted |

Only add another LLM judge if the **dominant** category still fails after a simple rule.

### 4.3 Critical bug found: cache early-return in `find_existing`

**Symptom:** `/memory` showed **two current** theme facts (light + dark) for the same user.

**Proven with Mem0 inspection:**

- Both had `fact_key = ui_theme` (not different keys).  
- Idle consolidate **merged** light only (`semantic_reinforced`) and left dark `current`.  
- Logs: cache returned `[light]` only; Mem0 had `[light, dark]`.

**Empty-cache experiment:** with `_recent` empty, `find_existing` returned both → supersede-all → **one** current. That confirmed the **cache-only early return** path.

**Fix applied:** `find_existing` always **unions session cache + Mem0** same-key currents before merge/supersede. `/memory` also collapses by `fact_key` via authority.

### 4.4 Thread vs fact (ChatGPT-style)

Soft prompts alone were not enough (e.g. Slack saved, old Teams thread still won). We added **code-level conflict resolution** for known preference topics, plus optional brief acknowledge of the losing side.

This is **answer-time** logic; it does not replace supersede on write.

### 4.5 Evidence gate (new)

Before promote, candidate **value** must appear in episode evidence / summary / trajectory (heuristic, MY+Latin friendly). Blocks invented durable writes.

Smoke tests: `tests/test_burmese_failure_smoke.py` (fast unit + optional `@pytest.mark.slow` LLM extract).

---

## 5. What I am doing now

**Stabilizing `memory-architecture-test` as a reliable baseline** (terminal harness + Mem0 playground).

Current focus:

1. **Verify** happy path after the `find_existing` fix (one current fact per key after flips).  
2. **Manual + automated smoke** for EN and Burmese (theme, manager, language, salary, etc.).  
3. **Document** architecture, conflict rules, and failure categories for the team.  
4. **Defer** for later (still inside this repo): agent-specific user facts, larger synonym lists, richer evidence checks.

**Working baseline when:**

- Remember → consolidate → one fact in `/memory`  
- Change value → supersede → still one current  
- Probe question matches `/memory`  
- Mixed Burmese + Latin still promotes when grounded  

---

## 6. Where data lives (this project only)

| Data | Where to look |
|------|----------------|
| Semantic facts | Mem0 Cloud (dashboard or API) — **not local Postgres / pgAdmin** |
| Episodes | `data/episodes/*.json` |
| Events | `data/events/*.jsonl` |
| Vectors | local Chroma `vector_store/` |
| Manual tests | `docs/manual-test-sentences.md`, `tests/test_burmese_failure_smoke.py` |
| Run harness | `python terminal_chat.py` |

---

## 7. Open / next (optional backlog in this playground)

- Agent-scoped personal facts (if HR vs Onboarding must not share all user prefs).  
- Grow synonym list from **logged** key misses only (keep LLM key-match as fallback).  
- Stronger evidence check only if MY↔EN paraphrase discards become common.  
- More EN/MY scenario coverage in `docs/manual-test-sentences.md`.

---

## 8. One-slide summary for discussion

> In `memory-architecture-test` we built layered memory (events → episodes → Mem0 facts) with cheap consolidate triggers, write-time supersede, and answer-time thread-vs-fact rules. Main finding: duplicate “current” facts were a **promotion lookup cache bug**, not “Burmese needs more LLMs.” Fix: always merge cache + Mem0 before supersede; add evidence grounding; categorize failures before adding judges. Current work: lock the EN/MY baseline in this playground.
