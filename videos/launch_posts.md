# PrismLib — Launch Post Pack

Three posts: Show HN, LinkedIn (personal), LinkedIn (company).
All numbers are from real Azure Container Apps runs (westus2):
- Cache 95.9% hit / Driver 98.6% latency reduction → Azure load test
- CHORUS mesh: token-sync, ~22ms cross-VNet frames, 633–674ms alerts,
  ~4s failover (97ms promotion) → 3-node Azure cluster, 2 VNets
- Only remaining caveat: same-region (cross-region untested)

---

## 1. Show HN (Hacker News)

**Title (≤80 chars):**
Show HN: PrismLib – semantic LLM cache + cluster mesh that cuts token spend

**URL field:** https://github.com/insightitsGit/prismlib

**Body (post as a comment immediately after submitting):**

Hi HN. PrismLib is an open-source (Apache 2.0) Python library that attacks three
recurring costs in production LLM apps:

1. Repeat answers. A semantic cache catches paraphrased questions ("how do
   refunds work?" == "what's your refund policy?"), so you stop paying for the
   same answer twice. On an Azure Container Apps load test (50 users, 300s,
   ~7k queries) it held a 95.9% hit rate.

2. Oversized prompts. A cosine-similarity compressor keeps only the relevant
   retrieved chunks before the prompt is built — 58–64% fewer context tokens in
   our tests, with no second model and no extra API call.

3. DB read latency. A streaming WAL driver serves reads from an in-process
   index instead of a network round-trip. On Azure: 142.83ms → 2.02ms avg
   (70.7x, 98.6% reduction) over an 11k-row index.

The part I'd most like feedback on is the cluster layer. Nodes share cached
answers over a small peer-to-peer protocol (CHORUS), and the same broadcast
traffic doubles as the failure detector — a warm standby that's been receiving
those frames all along promotes itself when the active node goes silent
(detection ~4s, actual promotion 97ms). No Raft, no external consensus store,
no Kubernetes operator. On a 3-node Azure deployment across two VNets,
cross-VNet frames averaged ~22ms and alerts propagated in 633-674ms.

Honesty up front: all numbers above are from real Azure Container Apps runs,
but both VNets are in one region (westus2), so cross-*region* latency and
partition behavior are untested. The whitepaper has a full "threats to
validity" section saying exactly this. The leaderless promotion also trades
Raft's correctness guarantees for simplicity; a partition could briefly produce
two actives — fine for caching workloads, not fine for a system of record.

Install: `pip install "prismlib[fabric]"`
Benchmark harness is in the repo (`benchmark/`) so you can reproduce or refute.

Would genuinely like to hear: where does the cache-traffic-as-failure-detector
idea break at scale? That's the design decision I'm least sure about.

---

## 2. LinkedIn — Personal page (attach the whitepaper PDF as a document post)

We were paying for the same AI answer five times. So I wrote a protocol to stop it.

If you run an LLM app as more than one server, here's a bill nobody warns you
about: every replica answers the same question independently. Five containers,
five identical "what's your refund policy?" calls, five times the cost.

I spent the last few months building PrismLib to kill that waste — and I just
wrote up the design and benchmarks as a whitepaper.

The idea I'm most proud of: the same messages nodes use to *share* cached
answers also work as the *failure detector*. When the active node goes quiet, a
warm standby that's been receiving those messages all along just takes over —
detection in ~4s, actual promotion in 97ms, no Kubernetes operator, no Raft
cluster, no human paged at 3am.

What's in the paper:
• Real Azure Container Apps numbers — 95.9% cache hit rate, 98.6% DB read
  latency reduction, ~22ms cross-VNet CHORUS frames, sub-700ms alerts
• How answers are shared across a cluster over a tiny protocol (CHORUS)
• A full "threats to validity" section — because the cluster ran in a single
  region and I'd rather tell you that than have you find out

That last part matters to me. It's a whitepaper, not a peer-reviewed study, and
I say so. If you've ever read a "scientific" product paper that was really a
brochure in a serif font — this isn't that.

Open source, Apache 2.0. Read it, run the benchmark, tell me where I'm wrong.

📄 Whitepaper attached. Library + reproducible benchmark in the first comment.

#LLM #RAG #AIinfrastructure #opensource #MLOps

First comment:
Library: https://pypi.org/project/prismlib · Source + benchmark:
https://github.com/insightitsGit/prismlib — `pip install "prismlib[fabric]"`.
Happy to answer anything technical here.

---

## 3. LinkedIn — Company page (attach PDF or the full_flow.svg diagram)

Cut your LLM cluster's token bill. No Redis. No Prometheus. No Kubernetes operator.

PrismLib is an open-source library (Apache 2.0) that removes the three biggest
hidden costs in production AI apps 👇

🔁 Stop paying for repeat answers.
A semantic cache catches paraphrased questions. 95.9% hit rate on an Azure load
test. Across a cluster, one node answers and every other node gets it free.

📉 Send smaller prompts.
Context compression keeps only the chunks relevant to each question — 58–64%
fewer context tokens per call, with no second model and no extra API call.

⚡ Faster reads.
A streaming DB driver serves reads from an in-process index: 142ms → 2ms on
Azure (98.6% latency reduction).

🟢 Failover without the infra.
Blue/Green/Orange hot-standby promotes a warm node in ~3-4 seconds
automatically. Health alerts in under a second — without Prometheus or Datadog.

Three layers, one install:
• PrismCache — semantic LLM cache
• PrismDriver — streaming DB driver
• PrismLib Micro — the cluster mesh

    pip install "prismlib[fabric]"

Free forever for individuals and teams. Need Slack/PagerDuty alerts, escalation
chains, and durable Kafka/NATS transport? ChorusMesh is the commercial layer on
top.

⭐ Star it, try it, and tell us your cache hit rate.
🔗 Links below. Enterprise? → insightits.info@gmail.com

#AI #LLM #RAG #MLOps #OpenSource #AIinfrastructure #MachineLearning

First comment:
📦 PyPI: https://pypi.org/project/prismlib · 💻 GitHub:
https://github.com/insightitsGit/prismlib · 📄 Technical whitepaper attached above.

---

## Posting notes
- HN: submit the GitHub URL, then immediately post the body as the first comment.
  Best time: weekday 8-10am ET. Don't ask for upvotes. Reply fast to every comment.
- LinkedIn: upload the PDF as a NATIVE document (renders as a swipeable carousel,
  far more reach). Put links in the FIRST COMMENT, not the post body.
- Company page can use full_flow.svg (videos/diagrams/) as the post image — strong
  scroll-stopper.
