#!/usr/bin/env python3
"""
Sanity check: verify Opus 4.5 can answer individual factual questions without CoT.
"""

import json
import os
import asyncio
from anthropic import AsyncAnthropic
from response_cache import ResponseCache

# Load API key
if "ANTHROPIC_API_KEY" not in os.environ:
    key_path = os.path.expanduser("~/.anthropic_api_key_rr")
    try:
        with open(key_path, "r") as f:
            os.environ["ANTHROPIC_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        pass

client = AsyncAnthropic()
cache = ResponseCache("caches/sanity_check_cache.json")


async def check_single_fact(fact: dict, semaphore: asyncio.Semaphore) -> dict:
    """Test if Opus 4.5 can answer a single factual question without CoT."""

    question = fact["question"]
    correct_answer = fact["answer"]

    messages = [
        {
            "role": "user",
            "content": f"Answer immediately with just the number, nothing else.\n\n{question}",
        },
        {
            "role": "assistant",
            "content": "Answer:",
        },
    ]

    cache_key = {"model": "claude-opus-4-5-20251101", "messages": messages}
    cached = await cache.get(cache_key)

    async with semaphore:
        if cached:
            response_text = cached["response"]
            print(f"[CACHED] {question[:50]}...")
        else:
            try:
                response = await client.messages.create(
                    model="claude-opus-4-5-20251101",
                    max_tokens=50,
                    messages=messages,
                    temperature=0.0,
                )
                response_text = response.content[0].text.strip()
                await cache.set(cache_key, {"response": response_text})
                print(f"[API] {question[:50]}...")
            except Exception as e:
                print(f"Error: {e}")
                return {"question": question, "correct": correct_answer, "predicted": None, "is_correct": False, "error": str(e)}

    # Parse response
    try:
        # Clean up response
        cleaned = response_text.strip().lower().replace(",", "")
        # Remove common prefixes
        for prefix in ["answer:", "the answer is", "there are", "there is", "it is", "it's"]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        predicted = int(cleaned.split()[0] if cleaned.split() else cleaned)
    except (ValueError, IndexError):
        predicted = None

    is_correct = predicted == correct_answer

    return {
        "question": question,
        "correct": correct_answer,
        "predicted": predicted,
        "response": response_text,
        "is_correct": is_correct,
    }


async def run_sanity_check(facts_file: str, max_facts: int = None):
    """Run sanity check on all facts."""

    # Load facts
    with open(facts_file, "r") as f:
        facts = json.load(f)

    if max_facts:
        facts = facts[:max_facts]

    print(f"Testing {len(facts)} facts...")

    # Create cache directory
    os.makedirs("caches", exist_ok=True)

    semaphore = asyncio.Semaphore(30)
    tasks = [check_single_fact(fact, semaphore) for fact in facts]
    results = await asyncio.gather(*tasks)

    await cache.save_cache(force=True)

    # Summarize
    correct = sum(1 for r in results if r["is_correct"])
    print(f"\n{'='*60}")
    print(f"Results: {correct}/{len(results)} correct ({100*correct/len(results):.1f}%)")
    print(f"{'='*60}")

    # Show errors
    errors = [r for r in results if not r["is_correct"]]
    if errors:
        print(f"\nIncorrect answers ({len(errors)}):")
        for r in errors:
            print(f"  Q: {r['question']}")
            print(f"     Expected: {r['correct']}, Got: {r['predicted']} (raw: {r.get('response', 'N/A')[:50]})")
            print()

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", "-f", default="static_facts.json", help="Facts file to test")
    parser.add_argument("--max", "-n", type=int, default=None, help="Max facts to test")
    args = parser.parse_args()

    asyncio.run(run_sanity_check(args.facts, args.max))
