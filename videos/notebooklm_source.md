# PrismLabPlusAPI — NotebookLM Source Document
## For video generation: architecture, proof, benchmarks, vision

---

## What This Is

PrismLabPlusAPI is an open-source Python library that solves a specific, measurable problem in production AI agent systems: every time an agent retrieves content from a knowledge base or API, it pays an embedding cost it does not need to pay.

This document covers what the library does, the real-world system it was proven in, the benchmark numbers, and how all the pieces connect. It is intended as source material for a technical explainer video.

---

## The Origin Story

The library was built by Amin Parva, founder of Insight IT Solutions LLC, while building a production AI agent platform for small and medium businesses. The platform serves 8 types of AI agents: restaurant, hotel, hospital, law firm, real estate, fitness, banking, and general business assistants.

Each agent handles real customer queries. A restaurant agent books reservations and answers menu questions. A hospital agent handles patient FAQs. A law firm agent explains services. These are not demos — they serve real customers.

While building this platform, a recurring cost appeared: every time an agent retrieved content from its knowledge base, it had to convert that content back into a vector before it could reason over it. The content was already a vector when it was stored. The conversion was redundant. At low volume this is invisible. At scale — across 8 agent types, multiple customers, hundreds of queries per hour — it becomes a measurable cost in both latency and dollars.

PrismLabPlusAPI was built to eliminate that redundant conversion.

---

## The Core Problem — The Embedding Tax

When an AI agent uses a knowledge base today, the flow looks like this:

Step 1: User asks a question. Step 2: Agent embeds the question into a float32 vector — this is necessary, there is no way around it. Step 3: Agent searches the knowledge base using that vector — fast, this is just a matrix operation. Step 4: Knowledge base returns matching documents as text — JSON with title, body, metadata. Step 5: Agent embeds each returned document — this is the redundant step. Step 6: Agent now has vectors it can reason over.

Step 5 is the problem. The knowledge base already embedded those documents when they were indexed. It stored them as vectors. Then it converted them back to text to send over the API. Then the agent converted them back to vectors. The content went vector to text to vector. That round trip costs time and money.

At top_k=5 results per query, an agent makes 6 embedding API calls per search: 1 for the query, 5 for the results. PrismLabPlusAPI reduces that to 1. Always. The saving is structural, not a tunable parameter.

At 1,000 queries per hour with top_k=10: the standard approach makes 11,000 embedding API calls per hour. PrismLabPlusAPI makes 1,000. That is 10,000 calls saved per hour, every hour, automatically.

---

## The Solution — Three Connected Layers

PrismLabPlusAPI is not a single component. It is three layers that work together.

### Layer 1: CHORUS Fabric — The Wire Protocol

CHORUS is a binary communication protocol designed specifically for float32 vector data. Instead of serializing vectors as JSON text — which inflates their size by 4.45 times — CHORUS sends raw binary float32 frames.

A standard REST API sends 1,000 vectors as 33,400 kilobytes of JSON text. CHORUS sends the same 1,000 vectors as 6,019 kilobytes of binary data. That is 82% less bandwidth. Not an estimate — a measured result from a real transatlantic benchmark between Azure datacenters in Virginia and Frankfurt, across 7,766 consecutive transmissions.

The encryption mechanism is mathematically elegant: the cipher operation is a matrix multiply, which is the same operation every neural network already performs. This means the encryption adds zero milliseconds of overhead. Zero. Measured, not theoretical.

CHORUS is the subject of a USPTO provisional patent application, number 64/096,156, filed June 22, 2026.

### Layer 2: PrismProjector — The Shared Space

Different AI providers use different embedding models. OpenAI uses 1,536-dimension vectors. Google Gemini uses 768 dimensions. The local sentence-transformers model uses 384 dimensions. These are incompatible — you cannot compare a vector from one model against a vector from another.

PrismProjector solves this with a Johnson-Lindenstrauss projection. It takes any embedding regardless of its original dimension and projects it into a fixed target space, typically 64 or 128 dimensions. The projection matrix is seeded by a SHA-256 hash of a tenant identifier, making it deterministic and reproducible across machines and restarts.

The critical property: two providers using the same tenant identifier produce vectors in the same projected space, even if they used completely different embedding models. A consumer can query both providers and rank all results together without knowing or caring what model each provider used. This makes provider-agnostic search possible without standardizing on a single embedding model.

### Layer 3: PrismAPI — The Consumer Protocol

PrismAPI is the outward-facing layer. It defines the contract between a data provider and an AI agent consumer.

The provider side embeds content once at index time, projects to 64 dimensions via PrismProjector, and when queried, returns float32 vectors directly without converting back to text.

The consumer side embeds its query once, sends the query vector to the provider, receives float32 vectors back, and uses them directly with zero additional embedding calls.

The provider wraps its existing search handler with a single decorator line. The original REST API continues working unchanged. The CHORUS channel is added alongside it. Existing clients see no difference. New CHORUS-native agents get pre-projected vectors.

---

## Proof — The Real System

The library was built while operating a production AI agent platform: InsightitsAIAgent.

This platform runs a Flask backend with LangGraph state machines for multi-step agent reasoning. It uses pgvector — PostgreSQL with a vector extension — storing 64-dimensional embeddings with an HNSW index for cosine similarity search. It routes LLM calls across AWS Bedrock with Claude 3 Sonnet, Google Gemini 1.5 Flash, OpenAI GPT-4o-mini, and Anthropic Claude. It runs 8 vertical agent types each with dedicated knowledge bases. The RAG pipeline includes intent classification, retrieval, relevance grading, grounding validation, and hallucination checking. Structured telemetry tracks tokens, latency, cost, and retrieval source per request.

The system works. The answers are meaningful and fast. The retrieval quality is validated across diverse domains — restaurant menus, medical FAQs, legal service descriptions, real estate listings. This is not a benchmark corpus of 200 documents. It is a real multi-domain production system.

---

## The Benchmark Numbers — Real HTTP, Not Simulated

All benchmarks use a real HTTP server running as a separate process. The client fires real HTTP requests. Nothing is simulated in-process.

Setup: 200-document corpus, 50 queries, 3 trials each, all-MiniLM-L6-v2 embedding model, 64-dimensional JL projection.

Embedding calls: Baseline makes 900 total calls for 50 queries across 3 trials, which is 6 per query. PrismAPI makes 150 total calls, which is 1 per query. That is 750 calls saved — an 83.3% reduction.

End-to-end latency: Baseline mean is 43.2 milliseconds composed of 6.4ms network round trip plus 36.8ms of client re-embedding. PrismAPI mean is 1.8 milliseconds of network round trip with zero re-embedding. That is 24 times faster true end-to-end.

Tail latency under concurrent load with 10 agents and 100 simultaneous requests: Baseline P99 is 517 milliseconds because the embedding model saturates. PrismAPI P99 is 6.6 milliseconds because there is no embedding model to saturate on the CHORUS path. That is 78 times better at the 99th percentile.

Wire bytes per query: Baseline sends 2,448 bytes as JSON text. PrismAPI sends 2,146 bytes as binary float32. That is a 12.3% reduction.

Retrieval quality at 64 dimensions versus full 384-dimension baseline: Recall at top 10 is 73%. The production usage pattern is to over-fetch twice the results and re-rank. At 128 dimensions, Recall at top 10 reaches approximately 91%.

Transatlantic CHORUS benchmark: Route was US East Virginia to Germany West Frankfurt on Azure Container Instances. 7,766 consecutive transmissions across 13 benchmark runs. P50 round-trip latency was 179 milliseconds, matching the physical minimum for the US-EU distance. Cipher overhead was 0 milliseconds. Watermark verification rate was 100%.

---

## The LangGraph Integration

LangGraph is the leading framework for building multi-step AI agent workflows. PrismLabPlusAPI ships a native LangGraph node that drops into any existing graph.

Before PrismAPI, what every LangGraph pipeline does today: Agent A retrieves content and returns it as text. Agent B re-embeds that text. Agent C re-embeds it again. In a 5-agent pipeline with 10 results each, that is 50 embedding calls for content that started as vectors.

After PrismAPI: Agent A retrieves content and returns float32 vectors. Agent B uses those vectors directly. Agent C uses them directly. Same 5-agent pipeline: 5 embedding calls. One per agent for its own query. Zero for results.

The integration is one line in the graph definition. The node reads the query from the graph state, retrieves from the PrismAPI provider, and writes an APIResponse object back to state. Downstream nodes receive pre-projected float32 vectors. The node supports fallback functions if the CHORUS endpoint is unreachable, preserving existing REST behavior as a safety net. The node also supports async invocation for async graphs.

---

## Multi-Provider Fan-Out

One capability of PrismAPI with no equivalent in standard RAG tooling: querying multiple knowledge bases simultaneously and merging results.

A legal research agent might need to query a case law database, a regulatory database, and a firm's internal precedent database. The standard approach queries each sequentially, re-embeds results from each, and then attempts comparison — slow, and the results are in incompatible vector spaces.

The PrismAPI approach: all three providers use the same tenant identifier, so all their projected vectors are in the same 64-dimensional space. The agent queries all three in parallel using a thread pool. Wall-clock time equals the slowest provider, not the sum. Results arrive as float32 vectors already in the same space, ranked by a single cosine sort across all providers. Each result is tagged with its source provider. Latency equals the slowest provider. Zero re-embedding.

---

## Privacy-Preserving Federated Search

A feature that emerges naturally from the architecture and has no direct equivalent elsewhere.

Because the query travels as a float32 vector and not as text, the provider never sees the query text. Because results travel as float32 vectors and not as text, the consumer never sees the raw document content. Both parties operate in projected vector space only.

For regulated industries this matters: a hospital network where Hospital A can search Hospital B's clinical notes without either side exposing raw records. A law firm searching a client database without the database operator seeing which cases are being researched. A financial institution querying competitor data under a sharing agreement without revealing search intent.

The projected vector space acts as a natural privacy boundary. Neither side exposes raw content. The shared projected space defined by the common tenant identifier is the only thing that crosses the boundary.

---

## The Test Suite

The library ships with 38 automated tests that run in 0.21 seconds with no machine learning model required. A mock embedder generates deterministic vectors from text hashes, making every test reproducible and offline.

Tests cover schema round trips for CHORUSFrame payloads, provider projection pipeline, consumer query paths, retry logic and connection management, multi-provider merge and deduplication, and LangGraph node behavior including query key aliases, top_k override from state, error key behavior, fallback function invocation, and async invocation.

---

## What It Is Not

PrismLabPlusAPI is not a vector database. It does not store documents or manage indexes.

It is not an embedding model. It is a transport and projection layer that works on top of any embedder — OpenAI, Gemini, Cohere, or local sentence-transformers.

It is not a replacement for REST APIs. It adds a parallel CHORUS channel. Existing REST clients continue working without modification.

---

## Recommended Video Structure

Opening at 30 seconds: Show the standard RAG loop. Query, embed, search, get text back, embed again. Highlight the redundant step. Message: your agent is re-embedding content that was already a vector.

Problem statement at 60 seconds: The math. At top_k=10 and 1,000 queries per hour, 10,000 unnecessary embedding calls. At OpenAI pricing, real money. At scale, real latency. The P99 story — 517ms baseline versus 6.6ms PrismAPI under 10 concurrent agents.

The real system at 60 seconds: InsightitsAIAgent. 8 vertical agents. Production customers. Restaurant, hospital, law firm, real estate. PrismLib powering the retrieval layer. Works in practice, not just benchmarks.

How it works at 90 seconds: Three layers. CHORUS for the wire — binary, fast, verified, patent pending. PrismProjector for the shared space — any model, same output, tenant-isolated. PrismAPI for the contract — provider embeds once, consumer never re-embeds.

The numbers at 60 seconds: Four stats simultaneously. 83.3% fewer embedding calls. 24 times faster true end-to-end. P99 78 times better under concurrent load. 82% bandwidth savings proven transatlantic.

LangGraph demo at 60 seconds: Before — agent pipeline with re-embedding at every hop. After — one line of code, vectors flow through state. Multi-provider fan-out querying three sources simultaneously.

Closing at 30 seconds: Open source, Apache 2.0. pip install prismlib-plus. GitHub insightitsGit/prismlibplusapi. Built by Insight IT Solutions LLC.

---

## Key Quotes for Narration

"The provider embedded that document once. The agent re-embedded it on every single retrieval. PrismAPI makes that stop."

"CHORUS sends vectors as vectors — not as text pretending to be vectors. 82% less bandwidth. Proven across 7,766 transmissions between the US and Europe."

"One decorator line on the provider. One node in your LangGraph graph. 83% fewer embedding calls. Structural, not tunable."

"The query travels as a float32 vector. The results travel as float32 vectors. Neither party ever sees raw text. That is privacy-preserving retrieval by architecture, not by policy."

"P99 latency under concurrent load: baseline 517 milliseconds, PrismAPI 6.6 milliseconds. The embedding model saturates. The CHORUS path does not."

"Built while running a real AI agent platform. Proven across 8 verticals — restaurant, hospital, law firm, hotel, real estate, fitness, banking. Not a benchmark demo. A real system."

"Five agent hops, 10 results each. Standard approach: 50 embedding calls for content that started as vectors. PrismAPI: 5 calls. One per agent for its own query. Zero for results."

---

## Technical Specifications

Python 3.11 and above. License Apache 2.0. Package name prismlib-plus. Install command: pip install prismlib-plus. GitHub: github.com/insightitsGit/prismlibplusapi. Required dependency: numpy. Optional dependencies: sentence-transformers, langgraph, fastapi. Test suite: 38 tests, 0.21 seconds, no ML model required. CHORUS patent: USPTO Provisional Application 64/096,156. Inventor: Amin Parva. Company: Insight IT Solutions LLC, Mission Viejo, California. Website: www.insightits.com.

---

## Glossary

Embedding: Converting text into a list of numbers called a vector that captures its meaning. Used by all modern AI search systems.

Float32 vector: A list of 32-bit floating-point numbers. The native format for AI embeddings. Standard REST APIs convert these to text for transmission and then back to float32 on receipt — a pointless round trip that PrismAPI eliminates.

Johnson-Lindenstrauss projection: A mathematical technique that reduces high-dimensional vectors to lower dimensions while approximately preserving the distances between them. Proven by a theorem published in 1984. PrismProjector uses this to create a shared compact space from any embedding model.

CHORUS frame: PrismLabPlusAPI's binary message format. Carries float32 vectors directly with a header containing a key identifier, sequence number, watermark, frame type, and payload length.

Tenant isolation: Each organization uses a unique identifier called a tenant_id that seeds the projection matrix. Different organizations produce incompatible vector spaces by design — their data cannot be cross-searched without explicit sharing.

Recall at top 10: A retrieval quality metric measuring how many of the baseline's top-5 results appear in PrismAPI's top-10 over-fetched results. 73% means 73% of relevant results are in the retrieved set before re-ranking.

P99 latency: The latency at the 99th percentile. 99% of requests are faster than this number. The most important latency metric for production systems under load.

LangGraph: A Python framework for building multi-step AI agent workflows as directed graphs. PrismRetrieverNode is a drop-in node that adds zero-re-embed retrieval to any LangGraph graph.

RAG: Retrieval-Augmented Generation. The technique of searching a knowledge base before asking an LLM to generate an answer. PrismAPI makes the retrieval step faster and cheaper without changing the RAG architecture.
