# PrismLabPlusAPI — Improvement Roadmap

Honest assessment of what would move this from a strong technical library to a market-ready product.
Ordered by impact.

---

## 1. Run the real HTTP benchmark at 128-dim

**What:** Re-run `benchmark/api/run_real_benchmark.py --dim 128` and commit the results.

**Why it matters:** The 91% Recall@10 number at 128-dim currently only exists from the loopback benchmark, not the real HTTP setup. Every technical reviewer will ask "what is the retrieval quality under real conditions?" The 64-dim real HTTP result is 73% — acceptable but not strong. 128-dim real HTTP is the number that closes the quality argument for enterprise adoption. This is a one-command fix.

**Effort:** 15 minutes. Run the benchmark, update `prism/api/README.md` and `PrismAPI.md` with the real numbers.

---

## 2. Add before/after metrics to InsightitsAIAgent

**What:** Instrument the `rag_retrieve` node in the existing production system to log embedding call counts per request, then integrate PrismAPI and log the same metric. Publish the before/after comparison.

**Why it matters:** Self-benchmarked synthetic results (200 docs, 50 queries) are credible but not compelling. "We measured this in our own production system across 8 vertical agents and real customer queries" is a completely different conversation. This turns the library from a benchmark into a proven production tool with a real case study.

**Metrics to capture before integration:**
- Embedding calls per conversation (query embeds + result embeds)
- P95 retrieval latency
- Cost per 1,000 queries (tokens × provider rate)

**Effort:** 2–3 hours to instrument, 1 week of production data collection, 1 hour to write up.

---

## 3. Validate PrismResonance against a standard benchmark

**What:** Run PrismResonance re-ranking against cosine similarity re-ranking on the BEIR benchmark (a standard public retrieval benchmark). Report which wins and by how much.

**Why it matters:** "Wave mechanics" is not a peer-reviewed algorithm. Technical reviewers at Anthropic, Google, or NVIDIA will ask what paper it is based on, and the answer is nothing published. This is a credibility blocker for licensing conversations. There are two paths: (a) if PrismResonance outperforms cosine re-ranking on BEIR, publish those numbers — that is real validation; (b) if it does not, rename it to "weighted cosine re-ranker" which is honest and defensible.

**Effort:** 1–2 days. BEIR is publicly available, evaluation scripts exist.

---

## 4. Get one external user with a public statement

**What:** Find one AI engineering team outside Insight IT Solutions who will use PrismRetrieverNode in their LangGraph pipeline and share a one-sentence quote about what they observed.

**Why it matters:** Every technical product lives or dies on social proof. One real user saying "we reduced our embedding calls by 80% and our P99 dropped from 400ms to 12ms" is worth more than 100 pages of benchmarks. The library is good enough to deliver that experience today — it just needs someone external to say so publicly.

**Who to target:** LangGraph community on Discord, AI engineers on Twitter/X who are posting about RAG optimization, early-stage AI startups building vertical agents (same profile as InsightitsAIAgent).

**Effort:** Outreach, not code.

---

## 5. Pursue the privacy-preserving federated search use case with a regulated industry

**What:** Package the federated search capability (query as vector, results as vectors, neither party sees raw content) as a specific pitch for healthcare or legal. Write a one-page brief targeting that buyer. Reach out to 5–10 companies in that space.

**Why it matters:** This is the use case where PrismAPI has no competition. Standard RAG tools have no answer for "how do I semantically search across organizations without sharing raw data?" The technical architecture already solves it. It just needs to be sold as a compliance feature, not a performance feature. Compliance features have budget. Performance features require ROI calculations.

**Target buyers:** Hospital networks, regional healthcare systems, legal discovery platforms, financial data exchanges, government data sharing consortiums.

**Effort:** 1 day to write the brief, ongoing outreach.

---

## 6. Pursue CHORUS licensing conversations

**What:** The USPTO provisional patent (application 64/096,156) covers the CHORUS matrix-multiply cipher and watermark protocol. Prepare a one-page licensing brief and initiate conversations with AI infrastructure companies.

**Why it matters:** The patent is the strongest defensible asset right now. CHORUS's zero-overhead encryption is technically novel — no other agent communication protocol uses a matrix multiply as the cipher operation. The companies that run at the scale where this matters (Anthropic, NVIDIA for AI Factory, Microsoft Azure AI, LangChain) are the right targets for a non-exclusive license.

**What to prepare:** A one-page brief covering the patent, the transatlantic benchmark (7,766 transmissions, 0ms cipher overhead, 100% watermark verification), and a proposed licensing structure (non-exclusive, per-transmission royalty or flat annual fee).

**Effort:** 1 day to prepare materials, then outreach.

---

## 7. Request a security review of the CHORUS cipher

**What:** Have a cryptographer review the CHORUS cipher (`V_enc = V_raw @ K` where K is a QR-decomposed orthogonal matrix) and the HMAC-SHA256 watermark scheme. Document the threat model explicitly.

**Why it matters:** The cipher is clever and the zero-overhead property is real. But "has this been reviewed by a cryptographer?" is a question every serious enterprise security team will ask before approving a production deployment. The answer currently is no. A single review from a credible source — even a known cryptographer on a public forum — dramatically increases trust. This is especially important for the privacy-preserving federated search use case, where security is the entire pitch.

**What to document regardless:** The explicit threat model. What CHORUS protects against (passive eavesdropping, message tampering, replay attacks via sequence numbers) and what it does not protect against (a compromised key holder, quantum attacks). Honesty about the threat model is more credible than overclaiming.

**Effort:** Find the right reviewer. The review itself may take 1–2 days of their time.

---

## 8. Add `--dim 128` as the default in benchmarks and documentation

**What:** Change the default `target_dim` in `run_real_benchmark.py` and the README from 64 to 128. Lead with the 91% Recall@10 number, not the 73% number.

**Why it matters:** The current default leading number is 73% Recall@10 at 64-dim. That is the number a skeptical reviewer fixates on. At 128-dim, wire size is roughly equivalent to JSON (near-parity) and recall is ~91%. That is a much stronger story: "91% of relevant results, equivalent wire size, 83% fewer embedding calls, 24× faster end-to-end." Lead with that. Let 64-dim be the "bandwidth-constrained" option, not the default.

**Effort:** 2 hours. Run benchmark at 128-dim, update defaults and docs.

---

## Summary Table

| # | Improvement | Effort | Impact |
|---|-------------|--------|--------|
| 1 | Real HTTP benchmark at 128-dim | 15 min | High — closes the quality argument |
| 2 | Before/after metrics in InsightitsAIAgent | 3–5 hrs + 1 week data | Very high — real case study |
| 3 | Validate PrismResonance on BEIR | 1–2 days | High — credibility with technical reviewers |
| 4 | Get one external user | Outreach | Very high — social proof |
| 5 | Federated search pitch to regulated industry | 1 day + outreach | High — unique use case with budget |
| 6 | CHORUS licensing conversations | 1 day + outreach | High — strongest defensible asset |
| 7 | Security review of CHORUS cipher | Find reviewer | Medium — unblocks enterprise adoption |
| 8 | Default to 128-dim in docs | 2 hrs | Medium — better first impression |
