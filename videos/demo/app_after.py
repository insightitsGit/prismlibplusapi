"""
app_after.py — the SAME endpoint wrapped with PrismCache (Video 1, the "fix").

Run it: the first question pays, every semantically-similar repeat is free.
Requires: pip install prismlib google-generativeai  and  GEMINI_API_KEY in env.
"""

import os
import google.generativeai as genai
from prism import PrismCache

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

# Semantic cache: near-identical questions count as the same question.
cache = PrismCache(similarity_threshold=0.92)


def answer(question: str, context: str) -> str:
    cached = cache.get(question)
    if cached is not None:
        print("  CACHE HIT — billed tokens: 0")
        return cached

    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    resp = model.generate_content(prompt)
    print(f"  cache miss — billed tokens: {resp.usage_metadata.total_token_count}")
    cache.set(question, resp.text)
    return resp.text


if __name__ == "__main__":
    CONTEXT = "Acme refunds any order within 30 days, no questions asked."

    print("Q1 (first time — pays):")
    print(answer("What is your refund policy?", CONTEXT))

    print("\nQ2 (same intent, different words — FREE):")
    print(answer("How do refunds work here?", CONTEXT))

    print("\nQ3 (exact repeat — FREE):")
    print(answer("What is your refund policy?", CONTEXT))
