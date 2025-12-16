#!/usr/bin/env python3
"""
Generate age_facts.json from the Kaggle AgeDataset.

The dataset contains ~120k historical figures with birth/death years.
This script filters for well-known researchers, scientists, and intellectuals.

Usage:
    python create_age_dataset.py -n 50                    # Generate 50 age facts
    python create_age_dataset.py -n 50 --verify           # Verify ages with LLM
    python create_age_dataset.py -n 50 --max-q 50000      # Only people with Q < 50000
    python create_age_dataset.py -n 50 --sanity-check     # Run sanity check on existing file

Note: The CSV ages use naive (death_year - birth_year), which may be off by 1.
Use --verify to correct ages using an LLM that knows actual birth/death dates.
"""

import random
import csv
import json
import argparse
import asyncio
import os
import re
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

# Keywords for filtering to researchers/scientists/intellectuals
RESEARCHER_KEYWORDS = [
    "mathematician",
    "physicist",
    "chemist",
    "biologist",
    "astronomer",
    "scientist",
    "philosopher",
    "logician",
    "economist",
    "sociologist",
    "psychologist",
    "naturalist",
    "inventor",
    "engineer",
    "computer",
    "geologist",
    "botanist",
    "zoologist",
    "geneticist",
    "neuroscientist",
    "anthropologist",
    "archaeologist",
    "historian",
    "linguist",
    "polymath",
    "scholar",
    "professor",
    "researcher",
    "theorist",
]

# Occupations to include
RESEARCHER_OCCUPATIONS = [
    "Scientist",
    "Mathematician",
    "Philosopher",
    "Researcher",
]

# People to exclude (uncertain dates, model consistently wrong, etc.)
EXCLUDE_NAMES = [
    "fibonacci",  # Birth/death dates uncertain (c. 1170 – c. 1240-50)
    "james prescott joule",  # Model consistently gets wrong age
]


def load_age_dataset(filepath):
    """Load the Kaggle age dataset."""
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse Q number from Id (e.g., "Q23" -> 23)
            q_match = re.match(r"Q(\d+)", row.get("Id", ""))
            if not q_match:
                continue
            q_num = int(q_match.group(1))

            if (
                "Name" not in row
                or "Birth year" not in row
                or "Death year" not in row
                or "Age of death" not in row
                or row["Birth year"] == ""
                or row["Death year"] == ""
                or row["Age of death"] == ""
            ):
                continue

            entries.append(
                {
                    "id": row.get("Id", ""),
                    "q_num": q_num,
                    "name": row.get("Name", ""),
                    "description": row.get("Short description", ""),
                    "occupation": row.get("Occupation", ""),
                    "birth_year": int(row["Birth year"]),
                    "death_year": int(row["Death year"]),
                    "age_of_death": int(row["Age of death"]),
                }
            )
    return entries


def is_researcher(entry):
    """Check if the entry is a researcher/scientist/intellectual."""
    desc_lower = entry["description"].lower()
    occ_lower = entry["occupation"].lower()

    # Check keywords in description
    for kw in RESEARCHER_KEYWORDS:
        if kw in desc_lower:
            return True

    # Check occupation
    for occ in RESEARCHER_OCCUPATIONS:
        if occ.lower() in occ_lower:
            return True

    return False


def filter_candidates(entries, max_q=30000, min_q=250, min_age=21, max_age=100):
    """Filter entries to research-related figures."""
    candidates = []
    for entry in entries:
        if entry["birth_year"] < 1500:
            continue

        # Check exclusion list
        if entry["name"].lower() in EXCLUDE_NAMES:
            continue

        if not (min_age <= entry["age_of_death"] <= max_age):
            continue

        # Filter by Q number (fame proxy)
        if entry["q_num"] > max_q or entry["q_num"] < min_q:
            continue

        # # Must be a researcher/scientist
        # if not is_researcher(entry):
        #     continue

        candidates.append(entry)

    return candidates


async def sanity_check_ages(facts, model="claude-opus-4-5-20251101", concurrency=40):
    """Run sanity check: ask model for each age and compare."""
    semaphore = asyncio.Semaphore(concurrency)

    cache = ResponseCache("caches/sanity_check_ages_cache.json")

    async def check_one(fact):
        prompt = f"At what age did {fact['name']} die? Just respond with the number."
        messages = [{"role": "user", "content": prompt}]

        response_text = await query_llm(messages, semaphore, cache, model=model, max_tokens=20, temperature=0)
        try:
            model_age = int(response_text)
        except ValueError:
            model_age = -1

        return {
            "name": fact["name"],
            "expected": fact["answer"],
            "model_age": model_age,
            "correct": model_age == fact["answer"],
            "response": response_text,
        }

    results = await asyncio.gather(*[check_one(f) for f in facts])

    # Print results
    correct = sum(1 for r in results if r["correct"])
    print(f"\nSanity check: {correct}/{len(results)} correct ({100*correct/len(results):.1f}%)")

    # Show mismatches
    mismatches = [r for r in results if not r["correct"]]
    if mismatches:
        print("\nMismatches:")
        for r in mismatches:
            print(f"  {r['name']}: expected {r['expected']}, model said {r['response']}")

    return results


client = AsyncAnthropic()


async def query_llm(messages, semaphore, cache, model="claude-opus-4-5-20250514", max_tokens=20, temperature=0.0):
    """Query LLM with caching and semaphore for concurrency control."""
    cache_key = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}

    cached = await cache.get(cache_key)
    if cached:
        return cached["response"]
    else:
        async with semaphore:
            client = AsyncAnthropic()
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            response_text = response.content[0].text.strip()
            await cache.set(cache_key, {"response": response_text})
            return response_text


async def verify_ages_with_llm(entries, model="claude-opus-4-5-20251101", concurrency=40):
    """Verify ages using an LLM and correct if needed. Uses async with caching."""

    os.makedirs("caches", exist_ok=True)
    cache = ResponseCache("caches/age_verify_cache.json")
    semaphore = asyncio.Semaphore(concurrency)

    async def verify_one(entry):
        prompt = f"""What age was {entry['name']} when they died?
Just respond with the number, nothing else. If you're not sure or the person is still alive or fictional, respond with "UNKNOWN"."""

        messages = [{"role": "user", "content": prompt}]
        response_text = await query_llm(messages, semaphore, cache, model=model, max_tokens=20, temperature=0)

        other_prompts = [
            f"""How old was {entry['name']} when they died? Just respond with a number that is your best guess.""",
            f"""What was the age of {entry['name']} when they died? Just respond with a number corresponding to your best guess.""",
            f"""Please tell me how old {entry['name']} was when they died. Just respond with the number.""",
        ]

        all_response_text_other = []
        for other_prompt in other_prompts:
            messages_other = [{"role": "user", "content": other_prompt}]
            response_text_other = await query_llm(
                messages_other, semaphore, cache, model=model, max_tokens=20, temperature=1.0
            )
            all_response_text_other.append(response_text_other)

        try:
            llm_age = int(response_text)
            llm_age_other = [int(r) for r in all_response_text_other]
        except ValueError:
            print(f"  {entry['name']}: Skipping (LLM response: {response_text})")
            return None

        if len(set([llm_age, *llm_age_other])) > 1:
            print(f"  {entry['name']}: LLM responses differ: {llm_age} vs {llm_age_other}, discarding")
            return None

        # Use LLM age if it differs (LLM accounts for birth/death months)
        if llm_age != entry["age_of_death"]:
            use_llm = llm_age == (
                entry["age_of_death"] - 1
            )  # off by one occurs because dataset just uses death_year - birth_year
            print(
                f"  Mismatch! {entry['name']}: CSV={entry['age_of_death']}, LLM={llm_age}"
                + (" -> Using LLM" if use_llm else " -> Discarding datapoint")
            )

            if not use_llm:
                return None

            return {**entry, "age_of_death": llm_age}
        else:
            return entry

    results = await asyncio.gather(*[verify_one(e) for e in entries])
    await cache.save_cache(force=True)

    return [r for r in results if r is not None]


def create_age_facts(entries):
    """Convert entries to age_facts.json format."""
    facts = []
    for entry in entries:
        facts.append(
            {
                "question": f"At what age did {entry['name']} die?",
                "answer": entry["age_of_death"],
                "category": "Age at death",
                "name": entry["name"],
            }
        )
    return facts


async def async_main(args):
    """Async main function to handle LLM operations."""
    # Load dataset
    print(f"Loading {args.input}...")
    entries = load_age_dataset(args.input)
    print(f"Loaded {len(entries)} entries")

    # Filter to researchers
    candidates = filter_candidates(entries, max_q=args.max_q, min_q=args.min_q)
    print(f"Found {len(candidates)} researcher candidates (Q < {args.max_q}, Q > {args.min_q})")

    # Sort/select
    if args.sort_by == "fame":
        candidates.sort(key=lambda x: x["q_num"])
    else:
        rng = random.Random(args.seed)
        rng.shuffle(candidates)

    selected = candidates[: 2 * args.num]  # Oversample some to allow for later filtering

    print(f"Selected {len(selected)} initial entries")

    model = "claude-opus-4-5-20251101"

    # Verify with LLM if requested
    if args.verify:
        print("\nVerifying ages with LLM...")
        selected = await verify_ages_with_llm(selected, model=model)
        print(f"Verified {len(selected)} entries")
        selected = selected[: args.num]  # Trim to desired number
        if len(selected) < args.num:
            print(f"Warning: Only {len(selected)} entries remain after verification, less than requested {args.num}")
        else:
            print(f"Trimmed to {len(selected)} entries after verification")

    # Show selection
    print("\nSelected people :")
    for entry in selected[:10]:
        print(f"  Q{entry['q_num']}: {entry['name']} (maybe age {entry['age_of_death']})")
    print("  ...")
    for entry in selected[-10:]:
        print(f"  Q{entry['q_num']}: {entry['name']} (maybe age {entry['age_of_death']})")

    found_euler = False
    for entry in selected:
        if "Euler" in entry["name"]:
            found_euler = True
            print(f"Euler found! Q{entry['q_num']}, age {entry['age_of_death']}")

    if not found_euler:
        print("Euler NOT found in selection!")

    # Convert to age facts format
    facts = create_age_facts(selected)

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(facts)} age facts to {args.output}")

    # Run sanity check if requested
    if args.sanity_check:
        print("\nRunning sanity check...")
        await sanity_check_ages(facts, model=model)


def main():
    parser = argparse.ArgumentParser(description="Generate age facts dataset")
    parser.add_argument("-n", "--num", type=int, default=150, help="Number of age facts to generate")
    parser.add_argument("--input", "-i", default="AgeDataset-V1.csv", help="Input CSV file")
    parser.add_argument("--output", "-o", default="age_facts.json", help="Output JSON file")
    parser.add_argument("--max-q", type=int, default=30000, help="Maximum Q number (fame filter)")
    parser.add_argument("--min-q", type=int, default=5000, help="Minimum Q number (fame filter)")
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify ages with LLM (default: True, use --no-verify to disable)",
    )
    parser.add_argument(
        "--sanity-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run sanity check on generated file (default: True, use --no-sanity-check to disable)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for selection")
    parser.add_argument(
        "--sort-by",
        choices=["fame", "random"],
        default="fame",
        help="Sort by fame (Q number) or random selection",
    )
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
