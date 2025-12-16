#!/usr/bin/env python3
"""
Demo/sanity check script that prints random prompts.
Shows the exact JSON that would be sent to the API.
"""

import json
import random
import argparse
from prompt_builder import (
    load_questions,
    select_few_shot_questions,
    build_full_prompt,
)


def main():
    parser = argparse.ArgumentParser(description="Show random prompts for compositional questions")
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of random questions to show")
    parser.add_argument("--input", "-i", default="compositional_questions.json", help="Input file")
    parser.add_argument("--k-shot", "-k", type=int, default=10, help="Number of few-shot examples")
    parser.add_argument("--repeat", "-r", type=int, default=1, help="Repeat question N times")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed for question selection")
    parser.add_argument("--full", action="store_true", help="Show full prompt JSON (vs summary)")
    args = parser.parse_args()

    # Load questions
    questions = load_questions(args.input)
    print(f"Loaded {len(questions)} questions from {args.input}")

    # Select few-shot examples (using minimize_overlap)
    _, default_few_shot_indices = select_few_shot_questions(
        questions, k_shot=args.k_shot, seed=42
    )

    # Select random test questions
    rng = random.Random(args.seed)
    selected_idxs = rng.sample(list(range(len(questions))), min(args.num, len(questions)))

    print(f"\n{'='*70}")
    print(f"Showing {len(selected_idxs)} random test question(s)")
    print(f"{'='*70}")

    for idx, q_idx in enumerate(selected_idxs):
        print(f"\n--- Question {idx + 1} (index {q_idx}) ---")
        question = questions[q_idx]
        print(f"Type: {question.get('composition_type', 'unknown')}")
        print(f"Question: {question['question']}")
        print(f"Components:")
        print(f"  1. {question['component1']['description']} = {question['component1']['value']}")
        print(f"  2. {question['component2']['description']} = {question['component2']['value']}")
        print(f"Answer: {question['answer']}")


        if args.full:
            # Build and show full prompt
            messages = build_full_prompt(
                questions=questions,
                question_index=q_idx,
                default_few_shot_indices=default_few_shot_indices,
                repeat_problem=args.repeat,
                add_cache_control=False,  # Don't include cache control for demo
                verbosity=3,
            )
            print(f"\nFull API request messages:")
            print(json.dumps(messages, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
