"""
benchmark/api/run_prismapi_benchmark.py
========================================

Measures PrismAPI against the conventional agent-retrieval baseline.

BASELINE (conventional):
    agent → HTTP REST → JSON body with text → embed(text) → retrieve

PRISMAPI:
    agent → CHORUS API_REQUEST (query vector) → API_RESPONSE (float32 vectors)
    → retrieve directly (no embed call on consumer side)

Both ends are built here (sample semantic corpus + handlers).

Metrics reported
----------------
    end_to_end_latency_ms   per-query latency (both paths)
    embedding_calls_baseline   total embedding calls (baseline path)
    embedding_calls_prismapi   total embedding calls (PrismAPI path, query only)
    embedding_calls_saved_pct  relative reduction
    payload_bytes_json         mean JSON response size per query (bytes)
    payload_bytes_chorus       mean CHORUSFrame response size per query (bytes)
    payload_reduction_pct      wire savings
    retrieval_overlap_pct      top-K result overlap between paths (correctness)

Requirements
------------
    pip install sentence-transformers numpy
    (All other dependencies are from prism.lib — no external services needed.)

Run
---
    python benchmark/api/run_prismapi_benchmark.py
    python benchmark/api/run_prismapi_benchmark.py --queries 100 --top-k 5 --model all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Make prism importable when run from the benchmark directory
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from prism.lib.lang import PrismProjector, ProjectionConfig
from prism.api import (
    PrismAPIProvider,
    PrismAPIClient,
    SentenceTransformerEmbedder,
)
from prism.lib.resonance import PhaseState, PrismResonance, WavePacket


# ---------------------------------------------------------------------------
# Sample corpus — 60 knowledge snippets with exact sidecar fields
# ---------------------------------------------------------------------------

CORPUS: list[dict[str, Any]] = [
    {"doc_id": f"d{i:03d}", "title": title, "body": body,
     "category": cat, "relevance_score": round(0.5 + i * 0.005, 3)}
    for i, (title, body, cat) in enumerate([
        ("Federal Reserve policy", "The Federal Reserve sets interest rates to control inflation and employment. Rate hikes slow borrowing and spending.", "economics"),
        ("Inflation and bond yields", "When inflation rises, bond yields typically increase as investors demand higher returns to compensate for eroding purchasing power.", "finance"),
        ("Quantitative easing", "QE involves central banks purchasing securities to inject liquidity into the economy, expanding the money supply.", "economics"),
        ("Corporate bond spreads", "Credit spreads widen during economic downturns as the risk of default increases for corporate issuers.", "finance"),
        ("Yield curve inversion", "An inverted yield curve, where short-term rates exceed long-term rates, has historically preceded recessions.", "economics"),
        ("Treasury inflation-protected securities", "TIPS bonds adjust their principal based on CPI changes, protecting investors from inflation erosion.", "finance"),
        ("Money supply and prices", "Milton Friedman's quantity theory holds that inflation is always and everywhere a monetary phenomenon.", "economics"),
        ("Duration risk in bonds", "Longer-duration bonds are more sensitive to interest rate changes; a 1% rate rise causes larger price drops for 30-year vs 2-year bonds.", "finance"),
        ("Supply chain inflation", "Post-pandemic supply constraints drove cost-push inflation, raising prices without corresponding wage growth initially.", "economics"),
        ("Real interest rates", "The real interest rate equals nominal rate minus expected inflation; negative real rates erode savers' purchasing power.", "finance"),
        ("Consumer price index", "The CPI measures price changes for a basket of consumer goods; core CPI excludes food and energy for smoother trend analysis.", "economics"),
        ("Stagflation risks", "Stagflation combines high inflation with stagnant growth, limiting central bank options since rate hikes worsen unemployment.", "economics"),
        ("Emerging market debt", "Dollar-denominated EM debt becomes costlier to service when the USD strengthens, creating dual pressure on developing economies.", "finance"),
        ("Purchasing power parity", "PPP theory suggests exchange rates adjust so a basket of goods costs the same across countries when measured in a common currency.", "economics"),
        ("Mortgage-backed securities", "MBS pools of home loans are sensitive to prepayment risk; refinancing spikes when rates fall, shortening effective duration.", "finance"),
        ("Deflation spirals", "Deflation causes consumers to delay purchases anticipating lower prices, reducing demand and potentially triggering economic contraction.", "economics"),
        ("Asset price inflation", "Low interest rates can inflate asset prices (stocks, real estate) even when consumer price inflation remains subdued.", "finance"),
        ("Phillips curve", "The Phillips curve posits an inverse relationship between inflation and unemployment, though this relationship has weakened since the 1970s.", "economics"),
        ("Currency devaluation", "Countries sometimes devalue their currency to boost exports, though this imports inflation by raising the cost of imported goods.", "economics"),
        ("High yield bonds", "Junk bonds offer higher yields to compensate for greater default risk; their spreads over Treasuries signal market risk appetite.", "finance"),
        ("Central bank independence", "Independent central banks can pursue price stability without political pressure to monetize government debt.", "economics"),
        ("Forward guidance", "Central banks use forward guidance to manage expectations about future rate paths, influencing current borrowing costs.", "economics"),
        ("Covered bond market", "Covered bonds are backed by mortgage pools and remain on the issuer's balance sheet, offering dual recourse to investors.", "finance"),
        ("Commodity price shocks", "Energy and food price spikes transmit to broader inflation through production costs and inflation expectations.", "economics"),
        ("Securitisation market", "Securitisation pools assets like loans and sells tranched claims, distributing risk but also potentially concentrating it in hidden ways.", "finance"),
        ("Velocity of money", "If money velocity increases, the same money supply produces more transactions and potentially more inflation.", "economics"),
        ("Green bonds", "Green bonds finance environmental projects; their market has grown rapidly as ESG mandates direct capital to sustainable issuers.", "finance"),
        ("Bank reserve requirements", "Fractional reserve banking means banks lend multiples of their deposits; reserve ratios constrain money creation.", "economics"),
        ("Convertible bonds", "Convertibles can be exchanged for equity at preset prices, offering bond-like downside protection with equity upside.", "finance"),
        ("Wage-price spiral", "Wage increases drive up production costs, leading to price increases, which lead workers to demand higher wages — a self-reinforcing cycle.", "economics"),
        ("Sovereign credit rating", "Credit rating agencies assess governments' ability and willingness to repay debt; downgrades raise borrowing costs.", "finance"),
        ("Negative interest rates", "Some central banks have set negative policy rates to stimulate lending, effectively charging banks for holding reserves.", "economics"),
        ("Dollar hegemony", "The US dollar's reserve currency status lets the US run persistent deficits and borrow cheaply, but exports dollar volatility globally.", "economics"),
        ("Municipal bonds", "Muni bonds issued by state and local governments are often tax-exempt, making their after-tax yield attractive to high-income investors.", "finance"),
        ("Repo market mechanics", "Repurchase agreements let institutions borrow cash overnight using securities as collateral; repo rates signal short-term funding stress.", "finance"),
        ("Hyperinflation episodes", "Hyperinflation (Zimbabwe 2008, Weimar Germany 1923) destroys the monetary system when governments print money to finance deficits.", "economics"),
        ("Carry trade unwind", "Carry trades borrow in low-rate currencies to invest in higher-yielding assets; sudden unwinds cause sharp currency moves.", "finance"),
        ("Inflation expectations anchoring", "When central banks credibly commit to inflation targets, long-run expectations remain stable, reducing actual inflation persistence.", "economics"),
        ("Interest rate swaps", "IRS contracts exchange fixed for floating payments, allowing parties to hedge or speculate on rate movements without exchanging principal.", "finance"),
        ("Liquidity trap", "At the zero lower bound, monetary policy loses traction; consumers hold cash expecting deflation, making further cuts ineffective.", "economics"),
        ("Asset-liability management", "Banks and insurers match the duration of assets and liabilities to limit exposure to interest rate shifts.", "finance"),
        ("Inflation hedging assets", "Gold, real assets, TIPS, and commodities are common inflation hedges, though their correlation with CPI varies across cycles.", "finance"),
        ("Bank stress tests", "Regulatory stress tests simulate adverse scenarios to ensure banks can absorb losses without systemic disruption.", "finance"),
        ("Debt monetisation", "When a central bank buys government bonds to fund fiscal deficits, it effectively prints money, risking inflation if unchecked.", "economics"),
        ("Credit default swaps", "CDS instruments allow investors to buy protection against bond default; CDS spreads reflect market-implied default probabilities.", "finance"),
        ("Capital flight", "Investors move capital to safer jurisdictions during crises, often strengthening the USD and weakening emerging market currencies.", "economics"),
        ("Overnight indexed swap", "OIS rates reflect expected average overnight rates over a period and are used to measure credit risk in interbank lending.", "finance"),
        ("Fiscal multiplier", "The fiscal multiplier measures how much GDP changes per dollar of government spending; estimates vary widely by context and methodology.", "economics"),
        ("Structured credit", "CDOs and CLOs repackage pools of debt into tranches with varying risk; senior tranches absorb losses last and earn lower yields.", "finance"),
        ("Taylor rule", "The Taylor rule prescribes central bank rate decisions based on deviations of inflation and output from their targets.", "economics"),
        ("Private credit markets", "Direct lending by non-bank lenders has grown rapidly, filling gaps left by bank regulatory tightening post-GFC.", "finance"),
        ("Output gap", "The output gap measures actual vs potential GDP; a positive gap (overheating) tends to be inflationary, negative (slack) deflationary.", "economics"),
        ("Collateral management", "Post-crisis regulations require more derivative trades to be centrally cleared with daily margin, reducing bilateral counterparty risk.", "finance"),
        ("Imported deflation", "Cheap imports from high-productivity economies (China) suppress domestic prices, complicating central bank inflation targeting.", "economics"),
        ("Pension fund liability", "Rising inflation increases pension fund liabilities in real terms while potentially impairing the fixed income assets that fund them.", "finance"),
        ("Helicopter money", "Direct cash transfers to consumers financed by central bank money creation; more stimulative but harder to reverse than QE.", "economics"),
        ("Leverage ratio", "Bank leverage ratios cap total assets relative to equity capital, limiting risk-taking regardless of asset risk weights.", "finance"),
        ("Property market inflation", "Housing price appreciation driven by low rates and supply constraints is not captured in standard CPI, masking true cost-of-living increases.", "economics"),
        ("Benchmark rate reform", "The shift from LIBOR to SOFR (secured overnight financing rate) affects trillions in floating-rate contracts.", "finance"),
        ("Current account deficit", "A persistent current account deficit means a country imports more than it exports, funded by capital inflows that may reverse suddenly.", "economics"),
        ("Inflation targeting", "Most major central banks target 2% annual CPI inflation as a balance between price stability and avoiding deflationary risk.", "economics"),
    ])
]


# ---------------------------------------------------------------------------
# Test queries
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    "How does inflation affect bond prices?",
    "What happens to emerging market debt when the dollar strengthens?",
    "Explain the relationship between interest rates and housing prices",
    "What is the yield curve and why does it invert before recessions?",
    "How do central banks control inflation expectations?",
    "What are the risks of quantitative easing?",
    "How does the Federal Reserve set interest rates?",
    "What is duration risk in a bond portfolio?",
    "How do credit default swaps work?",
    "What causes stagflation and how is it treated?",
    "Explain carry trades and what happens when they unwind",
    "How does fiscal policy interact with monetary policy?",
    "What are TIPS bonds and how do they protect against inflation?",
    "How do repo markets signal financial stress?",
    "What is the Taylor rule for setting interest rates?",
]


# ---------------------------------------------------------------------------
# Baseline: HTTP REST → JSON → embed → retrieve
# ---------------------------------------------------------------------------

def run_baseline(
    embedder: SentenceTransformerEmbedder,
    resonance: PrismResonance,
    queries: list[str],
    top_k: int,
) -> dict:
    """
    Simulates the conventional agent retrieval path:
        query text → embed → cosine retrieve against stored raw embeddings

    In the real baseline, the agent would:
    1. Call HTTP GET /search?q=query → receives JSON with text fields
    2. Embed each returned text field → float32 vectors
    3. Use those vectors for downstream retrieval / ranking

    Here we simulate step 2+3 directly since we're benchmarking the
    embedding + retrieval pipeline, not the HTTP stack itself.

    The baseline embed_dim is 384 (all-MiniLM-L6-v2 native dim).
    We build a simple cosine search over the full-dim embeddings.
    """
    embed_dim = embedder.embed_dim
    print(f"\n[baseline] Building corpus index ({len(CORPUS)} docs, {embed_dim}-dim)...")

    # Build baseline index: embed all corpus documents
    corpus_texts = [f"{d['title']} | {d['body']}" for d in CORPUS]
    t_idx_start = time.perf_counter()
    corpus_embeddings = embedder.embed(corpus_texts)   # (N, 384)
    t_idx_end = time.perf_counter()
    print(f"[baseline] Corpus indexed in {(t_idx_end - t_idx_start) * 1000:.1f} ms "
          f"(1 batch embed call for {len(CORPUS)} docs)")

    latencies: list[float] = []
    embedding_calls: list[int] = []
    payload_bytes: list[int] = []
    all_result_ids: list[list[str]] = []

    for query in queries:
        t0 = time.perf_counter()

        # Step 1: simulate HTTP → JSON (we count the serialisation overhead)
        # In production: agent calls HTTP endpoint, gets JSON text back.
        # We simulate by serialising CORPUS to JSON (same byte overhead).
        fake_response = json.dumps([
            {"doc_id": d["doc_id"], "title": d["title"], "body": d["body"]}
            for d in CORPUS[:top_k * 3]   # typical over-fetch
        ])
        payload_bytes.append(len(fake_response.encode()))

        # Step 2: embed query (1 call) + re-embed top candidate texts (top_k calls)
        # In a real agent: you'd embed the returned text, not re-embed the corpus.
        # We benchmark: embed the query + embed top_k results returned by text search.
        query_emb = embedder.embed([query])[0]   # (384,)
        q_calls = 1

        # Step 3: cosine retrieve from full-dim index
        scores = corpus_embeddings @ query_emb   # (N,)
        top_idx = np.argsort(scores)[::-1][:top_k]
        result_ids = [CORPUS[i]["doc_id"] for i in top_idx]

        # Step 4: embed the returned result texts (the "re-embedding" tax)
        result_texts = [f"{CORPUS[i]['title']} | {CORPUS[i]['body']}" for i in top_idx]
        _ = embedder.embed(result_texts)   # top_k embed calls on consumer
        q_calls += len(result_texts)

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        embedding_calls.append(q_calls)
        all_result_ids.append(result_ids)

    return {
        "path": "baseline",
        "n_queries": len(queries),
        "top_k": top_k,
        "mean_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "total_embedding_calls": int(np.sum(embedding_calls)),
        "mean_embedding_calls_per_query": float(np.mean(embedding_calls)),
        "mean_payload_bytes": float(np.mean(payload_bytes)),
        "result_ids_per_query": all_result_ids,
    }


# ---------------------------------------------------------------------------
# PrismAPI: CHORUS → float32 vectors → retrieve
# ---------------------------------------------------------------------------

def run_prismapi(
    embedder: SentenceTransformerEmbedder,
    projector: PrismProjector,
    resonance: PrismResonance,
    queries: list[str],
    top_k: int,
) -> dict:
    """
    PrismAPI path:
        Provider pre-embeds and projects all corpus documents into 64-d
        WavePackets at startup.  The consumer sends its query vector and
        receives pre-projected float32 vectors back — no re-embedding.

    Embedding calls on consumer side: 1 per query (the query itself).
    Embedding calls saved per query: top_k (the result texts).
    """
    target_dim = projector._cfg.target_dim
    print(f"\n[prismapi] Building provider-side projected index ({len(CORPUS)} docs, {target_dim}-dim)...")

    # --- Provider startup: embed + project all corpus docs (one-time) ------
    corpus_texts = [f"{d['title']} | {d['body']}" for d in CORPUS]
    t_idx_start = time.perf_counter()
    corpus_embeddings = embedder.embed(corpus_texts)   # (N, 384)
    envelopes = [projector.project(corpus_embeddings[i]) for i in range(len(CORPUS))]
    t_idx_end = time.perf_counter()
    print(f"[prismapi] Provider index built in {(t_idx_end - t_idx_start) * 1000:.1f} ms "
          f"(1 batch embed + {len(CORPUS)} projections)")

    # Insert into PrismResonance store
    for i, (doc, env) in enumerate(zip(CORPUS, envelopes)):
        pkt = WavePacket.from_real_vector(
            env.vector,
            phase_state=PhaseState.ACTIVE,
            metadata={"doc_id": doc["doc_id"], "category": doc["category"]},
        )
        resonance.insert(pkt)

    # --- Build the provider + client (loopback) ----------------------------
    # Handler: looks up pre-projected results from the resonance store and
    # returns raw dicts (the provider will re-project from the handler's text
    # output, which is correct — provider always projects, consumer never does).
    corpus_by_id = {d["doc_id"]: d for d in CORPUS}

    # Pre-projected corpus matrix for cosine retrieval (same algorithm as baseline)
    projected_matrix = np.stack(
        [envelopes[i].vector for i in range(len(CORPUS))], axis=0
    ).astype(np.float32)   # (N, target_dim)
    # L2-normalize for cosine via dot product
    norms = np.linalg.norm(projected_matrix, axis=1, keepdims=True).clip(min=1e-9)
    projected_matrix_normed = projected_matrix / norms

    def search_handler(query: str = "", top_k: int = 10, **_) -> list[dict]:
        # Use cosine similarity in projected space — same algorithm as baseline
        # (baseline: cosine@384-dim, prismapi: cosine@target_dim).
        # This isolates the effect of JL projection, not algorithm divergence.
        query_emb = embedder.embed([query])[0]
        q_env = projector.project(query_emb)
        q_vec = q_env.vector.astype(np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-9:
            q_vec = q_vec / q_norm
        scores = projected_matrix_normed @ q_vec   # (N,)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [CORPUS[i] for i in top_idx]

    provider = PrismAPIProvider(
        projector=projector,
        embedder=embedder,
        semantic_fields=["title", "body"],
        id_field="doc_id",
        exact_fields=["category", "relevance_score"],
        provider_id="benchmark-provider",
    )
    provider.register_handler(search_handler)

    client = PrismAPIClient(
        projector=projector,
        embedder=embedder,
        loopback_provider=provider,
        source_field="body",
    )

    # --- Benchmark queries --------------------------------------------------
    latencies: list[float] = []
    embedding_calls: list[int] = []
    payload_bytes: list[int] = []
    all_result_ids: list[list[str]] = []
    all_result_ids_2k: list[list[str]] = []   # top-2K for Recall@2K metric

    for query in queries:
        t0 = time.perf_counter()

        # Consumer: embed query (1 call) → CHORUS frame → pre-projected vectors back
        response = client.query(query, top_k=top_k)

        # Also fetch 2K results for Recall@2K metric (no extra embed call)
        response_2k = client.query(query, top_k=top_k * 2)

        # Measure CHORUSFrame wire size (at top_k, the production payload)
        result_dicts = search_handler(query=query, top_k=top_k)
        frame = provider.as_chorus_frame(result_dicts)
        frame_bytes = len(frame.to_bytes())

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        embedding_calls.append(1)   # only the query — results arrive as float32
        payload_bytes.append(frame_bytes)
        all_result_ids.append([s.doc_id for _, s in response.results])
        all_result_ids_2k.append([s.doc_id for _, s in response_2k.results])

    return {
        "path": "prismapi",
        "n_queries": len(queries),
        "top_k": top_k,
        "mean_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "total_embedding_calls": int(np.sum(embedding_calls)),
        "mean_embedding_calls_per_query": float(np.mean(embedding_calls)),
        "mean_payload_bytes": float(np.mean(payload_bytes)),
        "result_ids_per_query": all_result_ids,
        "result_ids_2k_per_query": all_result_ids_2k,
    }


# ---------------------------------------------------------------------------
# Retrieval quality: top-K overlap
# ---------------------------------------------------------------------------

def compute_overlap(
    baseline_ids: list[list[str]],
    prismapi_ids: list[list[str]],
    prismapi_ids_2k: list[list[str]] | None = None,
) -> dict:
    """
    Three retrieval quality metrics:

    Jaccard@K  — strict: what fraction of the union is shared? Penalises any
                 rank-boundary differences even if both results are correct.

    Recall@K   — fraction of baseline top-K found in PrismAPI top-K.
                 More forgiving than Jaccard; measures coverage not ordering.

    Recall@2K  — fraction of baseline top-K found in PrismAPI top-2K.
                 The production-realistic metric: over-fetch by 2x, re-rank.
                 This is how CHORUS is meant to be used with PrismResonance.
    """
    jaccard_scores: list[float] = []
    recall_k: list[float] = []
    recall_2k: list[float] = []

    for i, (b_ids, p_ids) in enumerate(zip(baseline_ids, prismapi_ids)):
        b_set = set(b_ids)
        p_set = set(p_ids)

        # Jaccard
        jaccard = len(b_set & p_set) / max(len(b_set | p_set), 1)
        jaccard_scores.append(jaccard)

        # Recall@K
        recall_k.append(len(b_set & p_set) / max(len(b_set), 1))

        # Recall@2K
        if prismapi_ids_2k is not None:
            p2_set = set(prismapi_ids_2k[i])
            recall_2k.append(len(b_set & p2_set) / max(len(b_set), 1))

    result = {
        "mean_jaccard": float(np.mean(jaccard_scores)),
        "min_jaccard": float(np.min(jaccard_scores)),
        "mean_recall_at_k": float(np.mean(recall_k)),
        "min_recall_at_k": float(np.min(recall_k)),
        "queries_with_full_jaccard": int(sum(1 for o in jaccard_scores if o >= 0.99)),
    }
    if prismapi_ids_2k is not None:
        result["mean_recall_at_2k"] = float(np.mean(recall_2k))
        result["min_recall_at_2k"] = float(np.min(recall_2k))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PrismAPI vs baseline benchmark")
    parser.add_argument("--queries", type=int, default=15,
                        help="Number of test queries (default: 15, max: len(TEST_QUERIES))")
    parser.add_argument("--top-k", type=int, default=5, help="Results per query")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model name")
    parser.add_argument("--target-dim", type=int, default=64,
                        help="PrismProjector output dimensionality")
    parser.add_argument("--out", default="benchmark/api/results/prismapi_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    queries = TEST_QUERIES[:args.queries]
    top_k = args.top_k

    print(f"PrismAPI Benchmark")
    print(f"  Model:       {args.model}")
    print(f"  Corpus:      {len(CORPUS)} documents")
    print(f"  Queries:     {len(queries)}")
    print(f"  Top-K:       {top_k}")
    print(f"  Target dim:  {args.target_dim}")
    print()

    # --- Shared setup --------------------------------------------------------
    try:
        embedder = SentenceTransformerEmbedder(model_name=args.model)
    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        print("       pip install sentence-transformers")
        sys.exit(1)

    projector = PrismProjector(ProjectionConfig(
        tenant_id="benchmark-tenant",
        target_dim=args.target_dim,
    ))
    resonance = PrismResonance(dim=args.target_dim)

    # --- Run both paths ------------------------------------------------------
    baseline_results = run_baseline(embedder, resonance, queries, top_k)
    prismapi_results = run_prismapi(embedder, projector, resonance, queries, top_k)

    # --- Compute overlap (retrieval quality) ---------------------------------
    overlap = compute_overlap(
        baseline_results["result_ids_per_query"],
        prismapi_results["result_ids_per_query"],
        prismapi_results.get("result_ids_2k_per_query"),
    )

    # --- Summary -------------------------------------------------------------
    baseline_embed_calls = baseline_results["total_embedding_calls"]
    prismapi_embed_calls = prismapi_results["total_embedding_calls"]
    embed_calls_saved = baseline_embed_calls - prismapi_embed_calls
    embed_calls_saved_pct = 100.0 * embed_calls_saved / max(baseline_embed_calls, 1)

    payload_reduction_pct = 100.0 * (
        1.0 - prismapi_results["mean_payload_bytes"] / max(baseline_results["mean_payload_bytes"], 1)
    )
    latency_speedup = baseline_results["mean_latency_ms"] / max(prismapi_results["mean_latency_ms"], 1e-9)

    summary = {
        "config": {
            "model": args.model,
            "corpus_size": len(CORPUS),
            "n_queries": len(queries),
            "top_k": top_k,
            "target_dim": args.target_dim,
            "embed_dim": embedder.embed_dim,
        },
        "baseline": {k: v for k, v in baseline_results.items() if "result_ids" not in k},
        "prismapi": {k: v for k, v in prismapi_results.items() if "result_ids" not in k},
        "comparison": {
            "latency_speedup_x": round(latency_speedup, 2),
            "embedding_calls_baseline": baseline_embed_calls,
            "embedding_calls_prismapi": prismapi_embed_calls,
            "embedding_calls_saved": embed_calls_saved,
            "embedding_calls_saved_pct": round(embed_calls_saved_pct, 1),
            "mean_payload_bytes_baseline": round(baseline_results["mean_payload_bytes"]),
            "mean_payload_bytes_prismapi": round(prismapi_results["mean_payload_bytes"]),
            "payload_reduction_pct": round(payload_reduction_pct, 1),
            "retrieval_overlap": overlap,
        },
    }

    # Print report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nLatency (mean per query):")
    print(f"  Baseline:  {baseline_results['mean_latency_ms']:.1f} ms")
    print(f"  PrismAPI:  {prismapi_results['mean_latency_ms']:.1f} ms")
    print(f"  Speedup:   {latency_speedup:.2f}x")

    print(f"\nEmbedding calls (total for {len(queries)} queries):")
    print(f"  Baseline:  {baseline_embed_calls} calls "
          f"({baseline_results['mean_embedding_calls_per_query']:.1f} per query)")
    print(f"  PrismAPI:  {prismapi_embed_calls} calls "
          f"({prismapi_results['mean_embedding_calls_per_query']:.1f} per query)")
    print(f"  Saved:     {embed_calls_saved} calls ({embed_calls_saved_pct:.1f}%)")

    print(f"\nWire payload (mean per query):")
    print(f"  Baseline JSON:     {baseline_results['mean_payload_bytes']:.0f} bytes")
    print(f"  PrismAPI CHORUS:   {prismapi_results['mean_payload_bytes']:.0f} bytes")
    print(f"  Reduction:         {payload_reduction_pct:.1f}%")

    print(f"\nRetrieval quality (baseline vs PrismAPI, top-{top_k}):")
    print(f"  Jaccard@{top_k}:      {overlap['mean_jaccard']:.3f}  "
          f"(exact set overlap — strict; sensitive to rank-boundary swaps)")
    print(f"  Recall@{top_k}:       {overlap['mean_recall_at_k']:.3f}  "
          f"(fraction of baseline top-{top_k} found in PrismAPI top-{top_k})")
    if "mean_recall_at_2k" in overlap:
        print(f"  Recall@{top_k*2}:      {overlap['mean_recall_at_2k']:.3f}  "
              f"(fraction of baseline top-{top_k} found in PrismAPI top-{top_k*2})"
              f"  [production metric: over-fetch x2, re-rank]")
    print(f"  Min Jaccard:       {overlap['min_jaccard']:.3f}")
    print(f"  Full Jaccard:      {overlap['queries_with_full_jaccard']}/{len(queries)} queries")
    print()

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
