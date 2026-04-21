"""
LLM API 
Author: Mohammad Anabtawi
Due: 4/22/2026



Requirements:
    pip install google-generativeai
"""

import google.generativeai as genai
import os

# ── Config ───────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=API_KEY)

MODEL = "gemini-1.5-flash"  # free tier model

# ── Two experiment inputs ────────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "id": 1,
        "label": "Creative / Open-ended",
        "system": "You are a witty sports journalist who specializes in combat sports.",
        "user":   "Write a short two-sentence hype intro for a Muay Thai championship match.",
    },
    {
        "id": 2,
        "label": "Analytical / Factual",
        "system": "You are a concise computer science tutor.",
        "user":   (
            "Explain the difference between a list and a tuple in Python. "
            "Keep it under 80 words."
        ),
    },
]

# ── Helper ───────────────────────────────────────────────────────────────────
def header(title):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")

def inspect_response(exp, response):
    header(f"Experiment {exp['id']}: {exp['label']}")

    print(f"\n[SYSTEM INSTRUCTION]\n{exp['system']}")
    print(f"\n[USER PROMPT]\n{exp['user']}")

    # Output text
    print(f"\n[OUTPUT TEXT]\n{response.text}")

    # Metadata
    print("\n[RESPONSE METADATA]")
    print(f"  Model           : {MODEL}")
    print(f"  Prompt tokens   : {response.usage_metadata.prompt_token_count}")
    print(f"  Response tokens : {response.usage_metadata.candidates_token_count}")
    print(f"  Total tokens    : {response.usage_metadata.total_token_count}")
    print(f"  Finish reason   : {response.candidates[0].finish_reason}")

    # Full candidates inspection
    print("\n[FULL CANDIDATE OBJECT]")
    print(response.candidates[0])

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\nLLM API Demo — Google Gemini")
    print(f"Using model: {MODEL}")

    for exp in EXPERIMENTS:
        model = genai.GenerativeModel(
            model_name=MODEL,
            system_instruction=exp["system"],
        )

        response = model.generate_content(exp["user"])
        inspect_response(exp, response)

    header("Done — both experiments complete.")

if __name__ == "__main__":
    main()
