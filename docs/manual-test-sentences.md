# Manual test sentences

Use **dave** / **eve** / **frank** for clean runs (startup options 4–6).  
Avoid alice / bob / carol if they still have old Mem0 leftovers.

Restart `terminal_chat.py` when switching agent or user.  
After remember / preference turns, run `/consolidate`, or say `thanks` / `bye`, or wait 5 minutes idle, or hit N-turn event count (HR 20 / Onboarding 15).  
No live topic-change trigger (extra LLM per message). Topic split still happens **inside** consolidate when it runs.  
Manual `/consolidate` is still the clearest for timed tests.

---

## How to run each scenario

1. Pick user + agent at startup
2. Prefer **new thread** for clean tests
3. Send the sentences
4. Run `/consolidate`
5. Restart and switch agent/user if needed
6. Ask the check question
7. Optionally open `/memory`

---

## 1. Sensitivity — English salary lock (HR only)

**Setup:** dave → HR Assistant → new thread

```text
Please remember: my current salary is $120437 per year.
```

```text
/consolidate
```

**Check A — Onboarding must NOT know**

Restart → dave → Onboarding → new thread

```text
What is my salary?
```

**Pass:** response must not contain `120437` / `$120437`

**Check B — HR still knows**

Restart → dave → HR → new thread

```text
What is my salary?
```

**Pass:** response can mention `120437`

---

## 2. Sensitivity — English public preference (shared)

**Setup:** dave → Onboarding → new thread

```text
Please remember: I prefer dark mode for all interfaces.
```

```text
/consolidate
```

**Check — HR should know**

Restart → dave → HR → new thread

```text
What is my UI preference?
```

**Pass:** mentions dark mode

---

## 3. Sensitivity — Burmese salary lock

**Setup:** dave → HR → new thread

```text
ကျေးဇူးပြုပြီး မှတ်ထားပေးပါ။ ကျွန်တော့် လစာက တစ်နှစ်ကို 120437 ဒေါ်လာပါ။
```

```text
/consolidate
```

**Check — Onboarding must NOT know**

Restart → dave → Onboarding → new thread

```text
ကျွန်တော့် လစာ ဘယ်လောက်လဲ။
```

Also try English:

```text
What is my salary?
```

**Pass:** no `120437`

---

## 4. Sensitivity — Burmese public preference (shared)

**Setup:** dave → Onboarding → new thread

```text
ကျေးဇူးပြုပြီး မှတ်ထားပေးပါ။ UI က dark mode ကို ပိုကြိုက်တယ်။
```

```text
/consolidate
```

**Check — HR should know**

Restart → dave → HR → new thread

```text
ကျွန်တော့် UI preference ကဘာလဲ။
```

Or:

```text
What is my UI preference?
```

**Pass:** mentions dark mode

---

## 5. Public notification preference (shared)

**Setup:** dave → Onboarding → new thread

```text
Please remember: I prefer Slack notifications, not email.
```

```text
/consolidate
```

**Check — HR**

Restart → dave → HR → new thread

```text
How should you notify me?
```

**Pass:** Slack, not email as the preference

---

## 6. Supersede same key (English)

**Setup:** dave → HR → new thread

Turn 1:

```text
Please remember: my notification preference is SlackChanTest01.
```

```text
/consolidate
```

Turn 2 (same user/agent; new or same thread is fine):

```text
Please remember: my notification preference is TeamsChanTest01.
```

```text
/consolidate
```

**Check:**

```text
What is my notification preference?
```

**Pass:** only `TeamsChanTest01` appears, not `SlackChanTest01`  
Also check `/memory` for one current notify fact.

---

## 7. Fact key consistency EN then Burmese

Goal: same preference should merge/supersede, not create two zombie facts.

**Setup:** dave → HR → new thread

English first:

```text
Please remember: I prefer Slack notifications. Use token SlackKeyTest77.
```

```text
/consolidate
```

Then Burmese update:

```text
ကျေးဇူးပြုပြီး မှတ်ထားပေးပါ။ notification က Teams ကို ပိုကြိုက်တယ်။ token က TeamsKeyTest77 ပါ။
```

```text
/consolidate
```

**Check:**

```text
How should you notify me?
```

**Pass:** `TeamsKeyTest77` wins  
**Also check `/memory`:** preferably one notify-related current fact, not both tokens as current

Write down both `fact_key` values if you can see them — they should match or be treated as the same concept.

---

## 8. User isolation (dave vs eve)

**Setup:** dave → HR → new thread

```text
Please remember: my UI theme is CoralThemeDave99.
```

```text
/consolidate
```

**Check — eve must not see it**

Restart → eve → HR → new thread

```text
What is my UI theme preference?
```

**Pass:** no `CoralThemeDave99`

---

## 9. Episode isolation (HR vs Onboarding)

**Setup:** dave → HR → new thread

```text
I am working on secret Project Quetzalcoatl99. Please discuss only with me in this HR chat.
```

(Have a short back-and-forth, then:)

```text
/consolidate
```

**Check — Onboarding should not recall that episode**

Restart → dave → Onboarding → new thread

```text
What project was I working on last time?
```

**Pass:** no `Quetzalcoatl99`  
(Semantic leak is separate; this checks episodic isolation.)

---

## 10. Retention — chatter dropped

**Setup:** dave → HR → new thread

English chatter only:

```text
ok
```

```text
thanks
```

```text
/consolidate
```

**Pass:** no useful new semantic fact from this

Burmese chatter:

```text
ဟုတ်
```

```text
ကျေးဇူး
```

```text
/consolidate
```

**Pass:** still no durable fact

Real Burmese fact (should keep):

```text
ကျွန်တော် Slack notification ပိုကြိုက်တယ်။ token က SlackKeep88 ပါ။
```

```text
/consolidate
```

**Pass:** fact can be retrieved later

---

## 11. Intent routing — procedural

**Setup:** dave → HR → new thread

English:

```text
How do I request PTO?
```

**Pass:** answer uses procedure / steps (Workday-style), not only a random personal fact

Burmese:

```text
PTO ဘယ်လို တောင်းရမလဲ။
```

**Pass:** still procedural / how-to style answer

---

## 12. Personal language policy

**Setup:** dave → HR → new thread

English:

```text
From now on reply to me in English when talking to me.
```

```text
/consolidate
```

**Check with Onboarding:**

Restart → dave → Onboarding

```text
Please greet me briefly.
```

**Pass:** English reply style if policy was promoted as public user fact

Burmese variant (separate run if needed):

```text
အခုကစပြီး မြန်မာလို ပြန်ပြောပေးပါ။
```

```text
/consolidate
```

**Check:** later agents reply in Burmese when asked something simple

---

## 13. Company handbook should not become user Mem0 fact

**Setup:** dave → HR → new thread

```text
What is the company leave notice rule for long leave?
```

(Let HR answer from KB.)

Then:

```text
Please remember that company policy for me.
```

```text
/consolidate
```

**Check `/memory`:**

**Pass:** should not create a personal semantic fact that copies the handbook as `user:dave` org policy  
Handbook may still appear via Knowledge Base on later questions — that is OK

---

## 14. Agent knowledge shared across users (optional)

If you already have an agent-scoped PTO formula fact / skill:

**dave → HR**

```text
How do I calculate my PTO balance?
```

**eve → HR**

```text
How do I calculate my PTO balance?
```

**Pass:** both can get the formula / procedure  
This is agent/shared procedural knowledge, not dave’s personal fact.

---

## Quick scorecard

| # | Scenario | User | Pass? | Notes |
|---|---|---|---|---|
| 1 | EN salary lock | dave | | |
| 2 | EN dark mode share | dave | | |
| 3 | MY salary lock | dave | | |
| 4 | MY dark mode share | dave | | |
| 5 | Slack notify share | dave | | |
| 6 | Supersede EN | dave | | |
| 7 | EN→MY key consistency | dave | | |
| 8 | User isolation | dave vs eve | | |
| 9 | Episode isolation | dave | | |
| 10 | Retention chatter | dave | | |
| 11 | Procedural routing EN/MY | dave | | |
| 12 | Language policy | dave | | |
| 13 | Handbook not user fact | dave | | |
| 14 | Agent knowledge shared | dave + eve | | |

---

## Tokens used (easy to spot leaks)

- `120437`
- `CoralThemeDave99`
- `Quetzalcoatl99`
- `SlackChanTest01` / `TeamsChanTest01`
- `SlackKeyTest77` / `TeamsKeyTest77`
- `SlackKeep88`

---

## Suggested user roles for this doc

| User | Use for |
|---|---|
| dave | Main happy-path + sensitivity tests |
| eve | Isolation partner vs dave |
| frank | Extra clean run if dave/eve get polluted |
| alice / bob / carol | Avoid unless you want to inspect old leftover data |
