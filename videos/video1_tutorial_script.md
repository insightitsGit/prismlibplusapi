# Video 1 — "Cut your LLM bill 76% in 5 lines" (Tutorial / Coding Demo)

**Target length:** 4:30
**Style:** Mixed — on-camera intro/outro, screen-record for the demo
**Audience:** Python devs building RAG / LLM apps who care about cost and latency

---

## SHOT LIST AT A GLANCE

| Time | Mode | What's on screen |
|------|------|------------------|
| 0:00–0:25 | 📷 Camera | You, hook |
| 0:25–1:00 | 🖥️ Screen | The problem — a normal RAG call + the bill |
| 1:00–2:45 | 🖥️ Screen | Live coding with Cursor/Claude — add PrismCache |
| 2:45–3:40 | 🖥️ Screen | Run it twice, show the token counter drop |
| 3:40–4:10 | 🖥️ Screen | One line to add the cluster layer |
| 4:10–4:30 | 📷 Camera | Outro + CTA |

---

## 📷 0:00–0:25 — HOOK (on camera)

> "Every time your AI app answers the same question twice, you pay twice.
> Most RAG apps re-embed, re-retrieve, and re-call the LLM for questions
> they've already answered. I'm going to show you how to stop that in about
> five lines of Python — and at the end, drop in one more line to share that
> cache across your whole cluster. Let's go."

*[Cut to screen]*

---

## 🖥️ 0:25–1:00 — THE PROBLEM (screen-record)

Show a plain RAG function. Keep it real — this is what everyone writes:

```python
# app.py — a normal RAG endpoint
import google.generativeai as genai

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

def answer(question: str, context: str) -> str:
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    resp = model.generate_content(prompt)
    return resp.text
```

**Narration:**
> "Here's a normal RAG endpoint. Nothing wrong with it — except watch what
> happens when two users ask the same thing."

Run it twice with the same question. Cut to the billing/token counter.

> "Same question, full price both times. Now multiply that by every FAQ,
> every 'what's your refund policy', every duplicate your users send. That's
> money on fire."

---

## 🖥️ 1:00–2:45 — LIVE CODING (screen-record, the star of the video)

Open Cursor (or Claude Code). Type this prompt **on camera** so viewers see the AI do the work:

> **Prompt to type:**
> "Wrap this RAG endpoint with PrismLib's PrismCache so identical and
> semantically-similar questions hit the cache instead of calling Gemini.
> Keep the same function signature."

Let the AI generate it. The result should look like:

```python
# app.py — now with PrismCache
import google.generativeai as genai
from prism import PrismCache

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

cache = PrismCache(
    similarity_threshold=0.92,   # treat near-identical questions as hits
)

def answer(question: str, context: str) -> str:
    cached = cache.get(question)
    if cached:
        return cached                       # <-- no LLM call, no cost

    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    resp = model.generate_content(prompt)
    cache.set(question, resp.text)
    return resp.text
```

**Narration while it generates:**
> "I'm letting Cursor wire it up. PrismCache isn't a dumb key-value cache —
> it's semantic. 'What's your refund policy' and 'how do refunds work' map to
> the same answer. That's the 92% similarity threshold right there."

> "Install is just `pip install prismlib`. One import, one object, two lines
> in the function. That's the whole integration."

---

## 🖥️ 2:45–3:40 — PROOF (screen-record)

Run the app. Fire the same/similar question 3–4 times. Show the token counter.

**On-screen overlay numbers (from real benchmarks):**
- First call: full tokens billed
- Repeat calls: **0 tokens** — served from cache
- Hit rate in production workloads: **91–96%**

**Narration:**
> "First call pays. Every repeat after that is free and instant. In our
> benchmarks PrismCache hits 91 to 96 percent on real workloads — that's
> 9 out of 10 calls you stop paying for."

---

## 🖥️ 3:40–4:10 — THE CLUSTER UPGRADE (screen-record)

> **Prompt to type in Cursor:**
> "Now make this cache shared across all nodes in my cluster using PrismLib's
> ClusterCache so a cache hit on one server serves every server."

```python
from prism.cluster import ClusterCache

cache = ClusterCache(
    node_id   = "green",
    peers     = ["blue:8000", "orange:8000"],
    threshold = 0.92,
)
# same .get() / .set() calls — now cluster-wide
```

**Narration:**
> "One swap — PrismCache becomes ClusterCache. Now when node A answers a
> question, node B and C already have it. Across a cluster that's another
> 76 percent token cut on top, because your servers stop duplicating each
> other's work. Same two lines in your function. You didn't change your logic."

---

## 📷 4:10–4:30 — OUTRO (on camera)

> "So: five lines for the local cache, one swap for the whole cluster.
> PrismLib is free and open source — `pip install prismlib`, link in the
> description. If you're running this at scale and want Slack alerts,
> PagerDuty, and failover, check out ChorusMesh on the same page.
> Drop a comment with your hit rate — I want to see it. See you in the next one."

*[End card: `pip install prismlib` · github.com/insightitsGit/prismlib]*

---

## ASSETS YOU NEED TO RECORD THIS
- [ ] `demo/app.py` — the before/after file (provided in `videos/demo/`)
- [ ] A Gemini API key set in env (real calls — no mocks)
- [ ] Terminal with a token/cost counter visible (or print token counts)
- [ ] Cursor or Claude Code open and signed in
- [ ] End card image: `pip install prismlib`

## ON-SCREEN TEXT / LOWER-THIRDS
- 0:30 — "A normal RAG endpoint"
- 1:10 — "pip install prismlib"
- 2:50 — "91–96% cache hit rate"
- 3:45 — "+76% token savings across the cluster"
- 4:15 — "Free & open source · Apache 2.0"
