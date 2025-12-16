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


def create_element_name_questions(
    static_facts,
    age_facts,
    atomic_facts,
    num_questions=100,
    seed=42
):
    """
    Create compositional questions that ask for element names.
    Example: "What is the name of the element with an atomic number equal to
             the sum of the age at which Euler died and the number of faces on a cube?"

    - Uses age_at_death and/or static facts (not atomic numbers)
    - Sum must be < 108 to have a valid element
    - Samples with replacement from static, without replacement from age
    """
    rng = random.Random(seed)

    # Build atomic number to element name mapping (only up to 108)
    atomic_number_to_element = {
        fact['answer']: fact['name'].lower()
        for fact in atomic_facts
        if fact['answer'] <= 108
    }

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

    questions = []
    already_used = set()
    used_age_indices = set()

    # Create a list for sampling
    age_indices = list(range(len(age_facts)))
    static_list = [format_static(fact) for fact in static_facts]

    max_attempts = num_questions * 100  # Prevent infinite loop
    attempts = 0

    while len(questions) < num_questions and attempts < max_attempts:
        attempts += 1

        # Choose composition type: age+age, age+static, or static+static
        # Ensure we sample without replacement from age (this is kinda random, claude misinterpreted one of my instructions)
        available_age_indices = [i for i in age_indices if i not in used_age_indices]

        if len(available_age_indices) < 2:
            # Not enough age facts left, use static+static only
            comp_type = "static+static"
        else:
            # ideally we wouldn't do age+age, but claude misinterpreted instruction and this is fine I guess. It usually doesn't work anyway...
            comp_type = rng.choice(["age+age", "age+static", "static+static"])

        if comp_type == "age+age":
            if len(available_age_indices) < 2:
                continue
            idx1, idx2 = rng.sample(available_age_indices, 2)
            desc1, val1 = format_age(age_facts[idx1])
            desc2, val2 = format_age(age_facts[idx2])
            used_age_indices.add(idx1)
            used_age_indices.add(idx2)
        elif comp_type == "age+static":
            if len(available_age_indices) < 1:
                continue
            idx1 = rng.choice(available_age_indices)
            desc1, val1 = format_age(age_facts[idx1])
            desc2, val2 = rng.choice(static_list)
            used_age_indices.add(idx1)
        else:  # static+static
            desc1, val1 = rng.choice(static_list)
            desc2, val2 = rng.choice(static_list)

        # Check if sum is valid (< 108 and corresponds to an element)
        answer_number = val1 + val2
        if answer_number >= 108 or answer_number not in atomic_number_to_element:
            continue

        # Check for duplicate component pairs
        sorted_descs = tuple(sorted([desc1, desc2]))
        if sorted_descs in already_used:
            continue
        already_used.add(sorted_descs)

        element_name = atomic_number_to_element[answer_number]

        question = f"What is the name of the element with an atomic number equal to the sum of {desc1} and {desc2}?"
        questions.append({
            "question": question,
            "answer": element_name,
            "answer_type": "element_name",
            "atomic_number": answer_number,
            "component1": {"description": desc1, "value": val1},
            "component2": {"description": desc2, "value": val2},
            "composition_type": comp_type
        })

    if len(questions) < num_questions:
        print(f"Warning: Could only generate {len(questions)} questions (requested {num_questions})")

    return questions


def main():
    parser = argparse.ArgumentParser(description="Create compositional questions dataset")
    parser.add_argument("--num", "-n", type=int, default=100, help="Number of questions to generate")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file")
    parser.add_argument("--element-names", action="store_true", help="Create element name questions instead of numerical questions")
    args = parser.parse_args()

    # Set default output file based on question type
    if args.output is None:
        args.output = "compositional_element_questions.json" if args.element_names else "compositional_questions.json"

    # Load all fact sources
    static_facts = load_facts("static_facts.json")
    age_facts = load_facts("age_facts.json")
    atomic_facts = load_facts("atomic_facts.json")

    print(f"Loaded {len(static_facts)} static facts")
    print(f"Loaded {len(age_facts)} age facts")
    print(f"Loaded {len(atomic_facts)} atomic facts")

    if args.element_names:
        # Create element name questions
        questions = create_element_name_questions(
            static_facts, age_facts, atomic_facts,
            num_questions=args.num,
            seed=args.seed
        )
        print(f"\nGenerated {len(questions)} element name questions")
    else:
        # Create standard compositional questions
        atomic_facts = [f for f in atomic_facts if f['answer'] > 20]
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
