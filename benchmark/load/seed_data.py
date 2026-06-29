"""
Seed data generator — 50k realistic customer support / business Q&A pairs.

The questions are organized into topic clusters so the cache can demonstrate
semantic similarity hits (paraphrased questions within the same cluster).

Topic distribution (mirrors real-world LLM usage patterns):
  30% — customer support (returns, shipping, billing)
  25% — product / technical FAQ
  20% — onboarding / account
  15% — data / analytics queries
  10% — general knowledge / reasoning

Each topic has ~20 question variants that are semantically similar but
worded differently — these generate cache hits.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Question clusters — each tuple is (canonical, [paraphrase1, paraphrase2, ...])
# ---------------------------------------------------------------------------

_CLUSTERS: list[tuple[str, list[str]]] = [
    # Customer support
    (
        "What is your return policy?",
        [
            "How do I return an item?",
            "Can I return something I bought?",
            "What are the return rules?",
            "How long do I have to return a product?",
            "Is there a return window?",
            "Can I get a refund?",
            "How do returns work?",
            "I want to return my order.",
            "What is the refund policy?",
            "Can I exchange instead of return?",
        ],
    ),
    (
        "When will my order ship?",
        [
            "How long does shipping take?",
            "What is the delivery time?",
            "When will I receive my package?",
            "How many days until my order arrives?",
            "Is my order on the way?",
            "What is the estimated delivery date?",
            "Why hasn't my order shipped yet?",
            "Can you check the status of my shipment?",
            "Where is my package?",
            "Has my order been dispatched?",
        ],
    ),
    (
        "How do I cancel my subscription?",
        [
            "I want to cancel my account.",
            "How do I stop my subscription?",
            "Where can I unsubscribe?",
            "How do I end my plan?",
            "Can I cancel anytime?",
            "Is there a cancellation fee?",
            "How to turn off auto-renewal?",
            "I want to stop my monthly plan.",
            "How do I downgrade my account?",
            "Can I pause my subscription instead?",
        ],
    ),
    (
        "How do I reset my password?",
        [
            "I forgot my password.",
            "Can't log in to my account.",
            "How do I recover my account?",
            "Password reset not working.",
            "I need a new password.",
            "How do I change my password?",
            "Locked out of my account.",
            "Send me a password reset link.",
            "How do I update my login credentials?",
            "My email for reset isn't working.",
        ],
    ),
    (
        "What payment methods do you accept?",
        [
            "Can I pay with PayPal?",
            "Do you accept credit cards?",
            "Is Apple Pay supported?",
            "Can I use a debit card?",
            "What currencies do you support?",
            "Do you accept crypto?",
            "Can I pay by invoice?",
            "Is bank transfer an option?",
            "Do you have buy-now-pay-later?",
            "Can I split the payment?",
        ],
    ),
    # Technical FAQ
    (
        "How do I integrate the API?",
        [
            "Where is the API documentation?",
            "How do I get started with the API?",
            "Show me a code example for the API.",
            "What is the base URL for the API?",
            "How do I authenticate with the API?",
            "What format does the API return?",
            "Is there a Python SDK?",
            "What are the API rate limits?",
            "How do I handle API errors?",
            "What version of the API should I use?",
        ],
    ),
    (
        "Why is my request returning a 401 error?",
        [
            "I'm getting an unauthorized error.",
            "Authentication is failing.",
            "My API key isn't working.",
            "Invalid credentials error.",
            "403 forbidden when calling the API.",
            "How do I fix an auth error?",
            "My token is expired.",
            "Bearer token not accepted.",
            "API returns unauthorized.",
            "How to refresh an expired token?",
        ],
    ),
    (
        "How do I increase my API rate limit?",
        [
            "I'm hitting the rate limit.",
            "Too many requests error.",
            "429 error from the API.",
            "How do I get higher throughput?",
            "Can I upgrade my quota?",
            "Rate limit exceeded, what should I do?",
            "How many requests per second am I allowed?",
            "Can I batch requests to avoid rate limits?",
            "I need more API capacity.",
            "How do I handle rate limiting in my app?",
        ],
    ),
    # Analytics / data
    (
        "How do I export my data?",
        [
            "Can I download my data as CSV?",
            "How do I get a data export?",
            "Is there a bulk export feature?",
            "How do I back up my data?",
            "Can I get my data in JSON format?",
            "How do I migrate my data to another system?",
            "Where can I download a report?",
            "How do I export all records?",
            "Is there an export API?",
            "Can I schedule automatic exports?",
        ],
    ),
    (
        "What are the SLA guarantees?",
        [
            "What is your uptime guarantee?",
            "What is the service level agreement?",
            "How reliable is the service?",
            "What is the promised availability?",
            "Is there a 99.9% uptime guarantee?",
            "What happens if you have downtime?",
            "Do you offer credits for outages?",
            "What is the maintenance window?",
            "How do you handle incidents?",
            "Where can I see your status page?",
        ],
    ),
    # Onboarding
    (
        "How do I invite team members?",
        [
            "Can I add other users to my account?",
            "How do I share access with my team?",
            "How do I create sub-accounts?",
            "Can multiple people use the same account?",
            "How do I set user roles?",
            "Can I restrict what my team can see?",
            "How do I remove a team member?",
            "Is there a team plan?",
            "How do permissions work?",
            "Can I transfer ownership of the account?",
        ],
    ),
    (
        "What is included in the free plan?",
        [
            "What can I do for free?",
            "What are the limits on the free tier?",
            "Is there a free trial?",
            "What features are available without paying?",
            "How long is the free trial?",
            "What happens after the free trial ends?",
            "Can I use this for free forever?",
            "What is the difference between free and paid?",
            "Is there an open source version?",
            "Can I test before buying?",
        ],
    ),
    # General / reasoning
    (
        "Explain quantum entanglement in simple terms.",
        [
            "What is quantum entanglement?",
            "How does quantum entanglement work?",
            "Can you explain entanglement simply?",
            "What does entangled mean in quantum physics?",
            "How are particles connected through entanglement?",
            "Is quantum entanglement faster than light?",
            "What is spooky action at a distance?",
            "Give me an example of quantum entanglement.",
            "Why is quantum entanglement important?",
            "What did Einstein say about entanglement?",
        ],
    ),
    (
        "What is machine learning?",
        [
            "Can you explain machine learning?",
            "How does machine learning work?",
            "What is the difference between AI and ML?",
            "Give me a simple definition of machine learning.",
            "What are examples of machine learning?",
            "How do machines learn from data?",
            "What is supervised learning?",
            "What is the difference between ML and deep learning?",
            "How is machine learning used in practice?",
            "What problems can machine learning solve?",
        ],
    ),
    (
        "What is the capital of France?",
        [
            "Where is Paris?",
            "What city is the capital of France?",
            "What country is Paris the capital of?",
            "Which European city is the capital of France?",
            "Name the capital city of France.",
            "Is Paris the capital of France?",
            "What is the main city in France?",
            "Where is the French government located?",
            "What city hosts the Eiffel Tower?",
            "Where is the Louvre museum?",
        ],
    ),
]


def get_seed_questions(n: int = 5000) -> list[str]:
    """
    Return n questions suitable for seeding the cache.
    ~50% are canonical, ~50% are paraphrases → produces ~50% hit rate on re-run.
    """
    questions: list[str] = []

    # First pass: canonical questions (cache misses — these prime the cache)
    for canonical, _ in _CLUSTERS:
        questions.append(canonical)

    # Second pass: paraphrases (these will be cache hits)
    paraphrases = [p for _, ps in _CLUSTERS for p in ps]
    random.shuffle(paraphrases)

    # Fill up to n by cycling through cluster paraphrases
    idx = 0
    while len(questions) < n:
        questions.extend(_CLUSTERS[idx % len(_CLUSTERS)][1])
        idx += 1

    return questions[:n]


def get_load_questions(n: int = 10000, hit_ratio: float = 0.7) -> list[str]:
    """
    Return n questions for load testing.

    hit_ratio: fraction that should be cache hits (paraphrased variants).
    The remaining fraction are novel questions that will be cache misses.
    """
    n_hits   = int(n * hit_ratio)
    n_misses = n - n_hits

    hits    = [p for _, ps in _CLUSTERS for p in ps] * (n_hits // len(_CLUSTERS) + 1)
    misses  = [
        f"Unique question {i}: describe the process of {_unique_topic(i)}"
        for i in range(n_misses)
    ]

    combined = hits[:n_hits] + misses[:n_misses]
    random.shuffle(combined)
    return combined


def _unique_topic(i: int) -> str:
    topics = [
        "converting a PDF to CSV",
        "setting up a Redis cluster",
        "calculating compound interest",
        "writing a regex for email validation",
        "deploying a Flask app to Heroku",
        "optimizing a PostgreSQL index",
        "building a REST API in Go",
        "parsing JSON in C#",
        "setting up SSL on Nginx",
        "configuring GitHub Actions",
    ]
    return topics[i % len(topics)] + f" (variant {i // len(topics)})"


def save_questions_json(path: str, n: int = 10000) -> None:
    """Save load-test questions to a JSON file for offline use."""
    questions = get_load_questions(n)
    with open(path, "w") as f:
        json.dump({"questions": questions, "count": len(questions)}, f, indent=2)
    print(f"Saved {len(questions)} questions to {path}")


if __name__ == "__main__":
    out = Path(__file__).parent / "questions.json"
    save_questions_json(str(out), n=10_000)
