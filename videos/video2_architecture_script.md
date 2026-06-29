# Video 2 — "How PrismLib Works: 3 Layers, One Mesh" (Architecture Explainer)

**Target length:** 4:45
**Style:** Mixed — on-camera intro/outro, screen-record/animated diagrams for the body
**Audience:** Eng leads & architects deciding whether to adopt PrismLib

---

## SHOT LIST AT A GLANCE

| Time | Mode | What's on screen |
|------|------|------------------|
| 0:00–0:30 | 📷 Camera | You, the big picture |
| 0:30–1:30 | 🖥️ Diagram | Layer 1 — PrismCache |
| 1:30–2:25 | 🖥️ Diagram | Layer 2 — PrismDriver |
| 2:25–3:30 | 🖥️ Diagram | Layer 3 — PrismLib Micro + CHORUS mesh |
| 3:30–4:15 | 🖥️ Diagram | The full request flow, end to end |
| 4:15–4:45 | 📷 Camera | Where ChorusMesh fits + outro |

Use the SVG diagrams in `videos/diagrams/` — reveal each block as you narrate it.

---

## 📷 0:00–0:30 — THE BIG PICTURE (on camera)

> "PrismLib is three layers that each kill a different kind of waste in an AI
> app. Layer one stops you paying for repeat questions. Layer two stops your
> database from being the bottleneck. Layer three stops your servers from
> duplicating each other's work. They stack — and they talk to each other over
> a tiny protocol called CHORUS. Let me show you how the whole thing fits."

*[Cut to architecture diagram]*

---

## 🖥️ 0:30–1:30 — LAYER 1: PrismCache (diagram: layer1.svg)

**Reveal:** request → PrismCache → (hit? return) / (miss? LLM)

**Narration:**
> "Layer one, PrismCache. Every request checks a semantic cache first. Not a
> string match — a *meaning* match. 'How do refunds work' and 'what's your
> refund policy' resolve to the same cached answer because they're 92 percent
> similar in embedding space."

> "On real workloads this hits between 91 and 96 percent. That means 9 out of
> 10 LLM calls never happen. No tokens, no latency, no cost. The miss falls
> through to your normal LLM call and gets cached on the way back."

**Key number on screen:** `91–96% hit rate`

---

## 🖥️ 1:30–2:25 — LAYER 2: PrismDriver (diagram: layer2.svg)

**Reveal:** app → PrismDriver (WAL stream) → database, with a fast-path arrow

**Narration:**
> "Layer two, PrismDriver. Your RAG app reads and writes context constantly —
> and a normal database driver makes you wait on every round trip. PrismDriver
> puts a write-ahead-log stream in front of the database, so reads serve from
> a hot in-memory layer and writes get streamed, not blocked on."

> "The result in our benchmarks is a 98.6 percent latency reduction on the
> hot path. Your retrieval stops waiting on disk."

**Key number on screen:** `98.6% latency reduction`

---

## 🖥️ 2:25–3:30 — LAYER 3: PrismLib Micro + CHORUS (diagram: layer3.svg)

**Reveal:** three nodes GREEN / BLUE / ORANGE connected by a CHORUS mesh

**Narration:**
> "Layer three is the cluster layer — PrismLib Micro. Now you've got many
> servers, and without coordination they each cache and compute the same
> things. Micro connects them with a mesh protocol called CHORUS."

> "CHORUS is seven tiny frame types — health, metrics, signals, cache-sync,
> and so on — that nodes broadcast to each other. When GREEN answers a
> question, it broadcasts the result, and BLUE and ORANGE already have it.
> That's another 76 percent token cut across the cluster."

> "It also gives you failover. GREEN is active, BLUE is a warm standby. If
> GREEN goes silent, BLUE promotes itself to active in about 3 to 4 seconds —
> no orchestrator, no human. And any node can raise an alert that reaches the
> whole cluster in under a second."

**Key numbers on screen:**
- `+76% token savings cluster-wide`
- `~3–4s automatic failover`
- `<1s alert propagation`
- `15–38ms CHORUS frame latency`

---

## 🖥️ 3:30–4:15 — THE FULL FLOW (diagram: full_flow.svg)

Walk one request through all three layers, end to end:

**Narration:**
> "So here's a single request, top to bottom. It hits PrismCache — 9 times
> out of 10 it's served right there, done. On a miss, it goes to PrismDriver,
> which serves context from the hot WAL layer instead of waiting on the
> database. Then the LLM is called once. The answer is cached locally — and
> broadcast over CHORUS so every other node in the cluster gets it for free.
> Three layers, one trip, almost no waste."

*[Animate the request flowing through each block, lighting up as it passes]*

---

## 📷 4:15–4:45 — CHORUSMESH + OUTRO (on camera)

> "Everything I just showed you is free and open source — Apache 2.0. The one
> paid piece is ChorusMesh: if you need Slack and PagerDuty alerts, escalation
> chains, and durable Kafka or NATS transport for the mesh, that's the
> commercial layer on top. But the core — caching, the fast driver, the
> cluster mesh, failover — costs nothing."

> "`pip install prismlib`, links in the description. If this helped, subscribe —
> next video I build a real RAG app on this from scratch. See you there."

*[End card: 3-layer diagram + `pip install prismlib`]*

---

## ASSETS YOU NEED
- [ ] `videos/diagrams/layer1.svg` … `full_flow.svg` (I'll generate these)
- [ ] Optional: a .pptx version if you want slide transitions instead of SVG reveals
- [ ] End card with the 3-layer stack

## CORE NUMBERS (keep these accurate on screen)
| Layer | Headline metric |
|-------|-----------------|
| PrismCache | 91–96% cache hit rate |
| PrismDriver | 98.6% latency reduction |
| PrismLib Micro | 76% token savings cluster-wide |
| Failover | ~3–4s automatic promotion |
| Alerts | <1s propagation |
| CHORUS latency | 15–38ms per frame |
