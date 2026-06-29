# PrismLib — A Plain-English Technical Introduction

> Source document for a NotebookLM Audio / Video Overview.
> Audience: software engineers and technically-curious people. Goal: explain
> what PrismLib is, why it exists, how it works, and show real code — without
> drowning the listener in jargon. Every number here is from a real benchmark.

---

## The one-sentence version

PrismLib is a free, open-source Python library that stops AI applications from
doing the same expensive work over and over — so they cost less and run faster.

## The problem, in everyday terms

Imagine a call center where five agents each answer the phone. A customer asks
"how do I reset my password?" Agent 1 looks it up and answers. A minute later a
different customer asks "I forgot my password, help" — and Agent 2, who has no
idea Agent 1 just answered almost the same thing, looks it up all over again.
Now imagine that happening thousands of times a day, and every lookup costs
money.

That's exactly what happens inside a typical AI app:

1. **It pays for the same answer twice.** Large language models charge by the
   "token" (roughly, by the word). When users rephrase the same question, most
   apps send each version to the model and pay full price every time.

2. **It sends bloated prompts.** AI apps usually stuff a big pile of background
   documents into every request "just in case" — and pay for all those words,
   even the irrelevant ones.

3. **It waits on the database.** Every time the app needs data, it makes a
   network round-trip to the database and waits.

PrismLib attacks all three. It's organized as three layers you can adopt
independently.

---

## Layer 1 — PrismCache: stop paying for repeat answers

PrismCache is a **semantic cache**. A normal cache only recognizes *exact*
repeats. "Semantic" means it recognizes *meaning* — so "how do refunds work?"
and "what's your refund policy?" are treated as the same question and share one
cached answer.

The mental model: it's a memory that sits in front of the language model. Ask
something it has effectively seen before, and it answers instantly for free.
Ask something genuinely new, and it calls the model once, then remembers.

Here's the entire integration — you wrap your existing model call:

```python
from prism.cache import PrismCache

cache = PrismCache.build(tenant_id="my-app", llm_model="gpt-4o")

def ask(question: str) -> str:
    return cache.get_or_call(
        query=question,
        call_fn=lambda: openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": question}],
        ).choices[0].message.content,
    )

# "How do I reset my password?" and "I forgot my password, help"
# return the SAME cached answer — only one call to the model.
```

**What it measured, for real:** in an Azure load test with 50 simultaneous
users sending about 7,000 questions, PrismCache served **95.9%** of them from
cache. That's roughly 19 out of every 20 model calls that never had to happen.

One nice detail: each customer ("tenant") gets a mathematically isolated cache,
so one company's data can never surface in another's results — not by a filter
that could be misconfigured, but by the math itself.

---

## Layer 2 — PrismDriver: stop waiting on the database

Normally your app asks the database a question and waits for the answer to come
back over the network. PrismDriver flips that around. A small companion process
sits next to the database and **streams changes** to your app as they happen, so
your app keeps a hot, up-to-date copy of the data *in its own memory*. Reads
never leave the process.

The analogy: instead of phoning the warehouse every time someone asks "is this
in stock?", you keep a live inventory screen on your own desk that updates
itself.

**What it measured:** on Azure, average read latency dropped from **142.8
milliseconds to 2.0 milliseconds** — about **70 times faster** — over an index
of 11,000 rows. And because the companion process watches the database's change
log, the local copy invalidates itself automatically; there's no stale data and
no manual cache-busting.

---

## Layer 3 — PrismLib Micro: share the savings across a whole cluster

Layers 1 and 2 help one server. But real apps run many copies (for capacity and
reliability), and those copies don't talk to each other — so they each
re-answer the same questions. Layer 3 fixes that.

PrismLib Micro connects your servers with a small peer-to-peer protocol called
**CHORUS**. When one server answers a question, it **broadcasts** the answer to
all the others. So the moment any server has answered something, every server
has it — for free.

```python
from prism.cluster.cache import ClusterCache

cache = ClusterCache(node_id="node-green", fabric=chorus_fabric)

answer = await cache.get_or_call(
    query=user_question,
    query_vector=embed(user_question),     # your embedder
    call_fn=lambda: llm.complete(...),     # your model call
    context_chunks=retrieved_docs,         # your retrieved documents
    chunk_vectors=doc_vectors,
)
# If ANY node in the cluster has answered this, it returns instantly,
# zero tokens billed — no Redis, no external cache server.
```

It also trims those bloated prompts (Layer 2 of the problem list). Before
calling the model, it ranks your background documents by relevance and keeps
only the most relevant few:

```python
# Out of 10 candidate context chunks, keep the 3 most relevant to THIS question.
# Measured: 58–64% fewer context tokens per call, with no second model call.
```

**What it measured** on a 3-node cluster deployed across two separate networks
on Azure:
- **76% average token savings** across the cluster
- Background documents trimmed by **58–64%** per request
- Servers stay in sync with messages crossing the network in about **22
  milliseconds**

---

## The clever part: failure handling comes for free

Here's the idea I'm most proud of, explained simply.

Any cluster needs a way to notice when a server dies and have a backup take
over. Normally that's a whole separate system — heartbeats, a coordination
service like Raft, sometimes a human getting paged.

But notice: in PrismLib, the servers are *already* constantly sending each other
those "here's a new answer" broadcasts. If a server goes quiet, the others
notice the silence. So **the same traffic that shares the cache also detects
failure.** A backup server — which has been receiving all those broadcasts, so
its memory is already warm and current — simply promotes itself and keeps
serving.

No separate heartbeat system. No external coordination service. The backup is
ready precisely *because* it was receiving cache updates all along.

**What it measured:** when the active server was silenced, a backup detected it
in about **4 seconds** (a deliberately cautious wait, so it doesn't overreact to
a brief hiccup) and completed the takeover in **97 milliseconds** — with no
human involved. Health alerts reached every server in **under 700 milliseconds**,
versus the 30–60 seconds a traditional monitoring system would take.

---

## The honest part (please keep this in the overview)

Good engineering content admits its limits, so:

- The cache and database numbers come from real load tests on Azure. The
  cluster numbers come from a real Azure deployment too, but it was **three
  servers in one region** — so behavior across continents, or at fifty servers,
  isn't proven yet.
- The "free failover" trades away the ironclad guarantees of a heavy
  coordination system for simplicity. In a rare network split, two servers
  could briefly both think they're in charge. That's totally fine for a *cache*
  (you just waste a little work) and would be wrong for, say, a bank ledger.
- The cache savings depend on how often your users actually repeat themselves.
  A support bot repeats constantly and saves a lot; a one-off creative tool
  repeats rarely and saves less.

This is why we publish a "threats to validity" section in the whitepaper instead
of just shouting the best number.

---

## Why it's built this way (the business model, briefly)

The core library — all three layers — is **free and open source** under the
Apache 2.0 license. You can read every line. A separate paid package called
**ChorusMesh** adds the things larger teams need: Slack and PagerDuty alerts,
escalation chains, and durable message transport over Kafka or NATS. Free code
builds trust and community; the paid layer is where the revenue is. (Same model
as Elastic or HashiCorp.)

Install is one line:

```bash
pip install "prismlib[fabric]"
```

---

## The 30-second recap (good for an intro or outro)

PrismLib makes AI apps cheaper and faster in three stackable steps: a semantic
cache that stops paying for repeat answers (95.9% hit rate), a streaming
database driver that kills read latency (70× faster), and a cluster layer that
shares answers across every server and — cleverly — reuses that same sharing
traffic to handle server failover automatically. It's free, open source, and
every number is from a reproducible benchmark.

Open source: github.com/insightitsGit/prismlib · Built by Insight IT Solutions.
