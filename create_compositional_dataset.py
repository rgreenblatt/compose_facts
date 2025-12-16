#!/usr/bin/env python3
"""
Create compositional questions by combining two factual questions.
E.g., "What is the sum of the atomic number of uranium and the age at which Euler died?"
"""

import json
import random
import argparse


def load_facts(filepath):
    """Load facts from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_compositional_questions(
    static_facts,
    age_facts,
    atomic_facts,
    num_questions=100,
    seed=42
):
    """
    Create compositional questions by combining pairs of facts.

    Composition types:
    1. atomic_number + age_at_death (like the Euler example)
    2. static_fact + atomic_number
    3. static_fact + age_at_death
    4. static_fact + static_fact
    """
    rng = random.Random(seed)


    # Helper to format fact into question component
    def format_atomic(fact):
        return f"the atomic number of {fact['name'].lower()}", fact['answer']

    def format_age(fact):
        return f"the age at which {fact['name']} died", fact['answer']

    def format_static(fact):
        # Convert "What is the number of X?" to "the number of X"
        q = fact['question']
        if q.startswith("What is the number of "):
            q = q[22:]  # Remove "What is the number of "
            if q.endswith("?"):
                q = q[:-1]
            return f"the number of {q}", fact['answer']
        # Legacy format: "How many X?"
        if q.startswith("How many "):
            q = q[9:]  # Remove "How many "
            if q.endswith("?"):
                q = q[:-1]
            return f"the number of {q}", fact['answer']
        return q, fact['answer']

    desc_vals_by_item = [
        [format_atomic(fact) for fact in atomic_facts],
        [format_age(fact) for fact in age_facts],
        [format_static(fact) for fact in static_facts],
    ]

    name_by_item = ["atomic", "age", "static"]

    questions = []
    # Generate questions ensuring diversity
    already_used = set()

    while len(questions) < num_questions:
        first_item = rng.randint(0, 2)
        second_item = rng.choice([i for i in range(3) if first_item == 2 or i != first_item]) # static can repeat
        desc1, val1 = rng.choice(desc_vals_by_item[first_item])
        desc2, val2 = rng.choice(desc_vals_by_item[second_item])

        sorted_descs = tuple(sorted([desc1, desc2]))
        if sorted_descs in already_used:
            continue
        already_used.add(sorted_descs)

        sorted_item_0, sorted_item_1 = tuple(sorted([first_item, second_item]))
        comp_type = f"{name_by_item[sorted_item_0]}+{name_by_item[sorted_item_1]}"

        answer = val1 + val2

        # Filter out answers > 999 (to stay in 0-999 range)
        if answer > 999:
            continue

        question = f"What is the sum of {desc1} and {desc2}?"
        questions.append({
            "question": question,
            "answer": answer,
            "component1": {"description": desc1, "value": val1},
            "component2": {"description": desc2, "value": val2},
            "composition_type": comp_type
        })

    return questions


def main():
    parser = argparse.ArgumentParser(description="Create compositional questions dataset")
    parser.add_argument("--num", "-n", type=int, default=100, help="Number of questions to generate")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--output", "-o", type=str, default="compositional_questions.json", help="Output file")
    args = parser.parse_args()

    # Load all fact sources
    static_facts = load_facts("static_facts.json")
    age_facts = load_facts("age_facts.json")
    atomic_facts = load_facts("atomic_facts.json")

    atomic_facts = [f for f in atomic_facts if f['answer'] > 20]

    print(f"Loaded {len(static_facts)} static facts")
    print(f"Loaded {len(age_facts)} age facts")
    print(f"Loaded {len(atomic_facts)} atomic facts")

    # Create compositional questions
    questions = create_compositional_questions(
        static_facts, age_facts, atomic_facts,
        num_questions=args.num,
        seed=args.seed
    )

    print(f"\nGenerated {len(questions)} compositional questions")

    # Show distribution by type
    type_counts = {}
    for q in questions:
        t = q['composition_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    print("\nDistribution by type:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")

    # Show examples
    print("\nExamples:")
    for q in questions[:5]:
        print(f"  Q: {q['question']}")
        print(f"     A: {q['answer']} ({q['component1']['value']} + {q['component2']['value']})")
        print()

    # Save
    with open(args.output, 'w') as f:
        json.dump(questions, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
