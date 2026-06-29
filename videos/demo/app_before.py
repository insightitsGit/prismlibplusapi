"""
app_before.py — a normal RAG endpoint (Video 1, the "problem" shot).

Run this twice with the same question and watch it pay full price both times.
Requires: pip install google-generativeai  and  GEMINI_API_KEY in your env.
NO mocks — these are real Gemini calls so the token counts on screen are real.
"""

import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")


def answer(question: str, context: str) -> str:
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    resp = model.generate_content(prompt)
    # Show the real billed tokens on camera
    usage = resp.usage_metadata
    print(f"  billed tokens: {usage.total_token_count}")
    return resp.text


if __name__ == "__main__":
    CONTEXT = "Acme refunds any order within 30 days, no questions asked."

    # Two users, same intent — full price BOTH times
    print("Q1:")
    print(answer("What is your refund policy?", CONTEXT))
    print("\nQ2 (same intent, different words):")
    print(answer("How do refunds work here?", CONTEXT))
