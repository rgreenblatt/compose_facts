#!/usr/bin/env python3
"""
Evaluation script for compositional questions using Claude Opus 4.5.
Tests whether the model can combine two facts and add them without chain-of-thought.
Uses 10-shot prompting with prefilled "Answer:" response.
"""

import json
import os
import asyncio
import argparse
from anthropic import AsyncAnthropic
from response_cache import ResponseCache
from prompt_builder import (
    build_full_prompt,
    load_questions,
    select_few_shot_question_for_problem,
    select_few_shot_questions,
    get_component_facts,
    build_user_message,
    build_few_shot_messages,
)

# Load API key
if "ANTHROPIC_API_KEY" not in os.environ:
    key_path = os.path.expanduser("~/.anthropic_api_key_rr")
    try:
        with open(key_path, "r") as f:
            os.environ["ANTHROPIC_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        pass

client = AsyncAnthropic()

async def evaluate_question(
    questions,
    question_index,
    few_shot_indices,
    semaphore,
    cache,
    model="claude-opus-4-5-20251101",
    repeat_problem=None,
    max_retries=8,
    verbosity=2,
):
    """Evaluate a single question."""
    max_tokens = 50

    messages = build_full_prompt(
        questions=questions,
        question_index=question_index,
        default_few_shot_indices=few_shot_indices,
        repeat_problem=repeat_problem,
        add_cache_control=True,
    )
    question = questions[question_index]

    cache_key = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    cached = await cache.get(cache_key)

    async with semaphore:
        if cached:
            response_text = cached["response"]
            if verbosity >= 3:
                print(f"[CACHED] Question {question_index + 1}")
        else:
            response_text = None
            # Make API call with retry logic
            for attempt in range(max_retries):
                if verbosity >= 2:
                    print(f"[API] Question {question_index + 1}" + (f" (attempt {attempt + 1})" if attempt > 0 else ""))

                try:
                    response = await asyncio.wait_for(
                        client.messages.create(
                            model=model,
                            max_tokens=max_tokens,
                            messages=messages,
                            temperature=0.0,
                        ),
                        timeout=120.0,
                    )
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt * 0.25
                        print(f"Error on question {question_index + 1} ({attempt=}), retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"Failed after {max_retries} attempts: {e}")
                        return {
                            "question_index": question_index,
                            "correct_answer": question["answer"],
                            "predicted_answer": None,
                            "is_correct": False,
                            "error": str(e),
                            "question": question["question"],
                        }

                response_text = response.content[0].text.strip()

                # Cache the response
                await cache.set(cache_key, {"response": response_text})
                break

            assert response_text is not None

        # Parse response
        try:
            cleaned = (
                response_text.strip()
                .lower()
                .replace(",", "")
                .removeprefix("answer:")
                .strip()
            )
            predicted = int(cleaned.split()[0] if cleaned.split() else cleaned)
        except (ValueError, IndexError):
            if verbosity >= 1:
                print(f"Could not parse response: {response_text[:50]}")
            predicted = None

        is_correct = predicted == question["answer"]

        return {
            "question_index": question_index,
            "correct_answer": question["answer"],
            "predicted_answer": predicted,
            "is_correct": is_correct,
            "response": response_text,
            "question": question["question"],
            "composition_type": question.get("composition_type"),
            "cached": cached is not None,
        }


async def run_evaluation(
    input_file,
    output_file=None,
    model="claude-opus-4-5-20251101",
    repeat_problem=None,
    concurrency=20,
    k_shot=10,
    verbosity=2,
):
    """Run evaluation on all questions."""
    # Load questions
    questions = load_questions(input_file)
    print(f"Loaded {len(questions)} questions from {input_file}")

    _, few_shot_indices = select_few_shot_questions(
        questions, k_shot=k_shot, seed=42
    )

    # Initialize cache (model-specific)
    cache_suffix = f"_{model.replace('-', '_')}"
    if repeat_problem and repeat_problem > 1:
        cache_suffix += f"_r{repeat_problem}"
    cache_file = f"caches/eval_compositional{cache_suffix}.json"
    os.makedirs("caches", exist_ok=True)
    cache = ResponseCache(cache_file)

    # Create semaphore and tasks
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        evaluate_question(
            questions, i, few_shot_indices, semaphore, cache,
            model=model, repeat_problem=repeat_problem, verbosity=verbosity
        )
        for i in range(len(questions))
    ]

    # Run evaluation
    results = await asyncio.gather(*tasks)

    # Save cache
    await cache.save_cache(force=True)

    # Calculate statistics
    correct = sum(1 for r in results if r["is_correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.1%}")

    # Breakdown by composition type
    type_stats = {}
    for r in results:
        t = r.get("composition_type", "unknown")
        if t not in type_stats:
            type_stats[t] = {"correct": 0, "total": 0}
        type_stats[t]["total"] += 1
        if r["is_correct"]:
            type_stats[t]["correct"] += 1

    print("\nBy composition type:")
    for t, stats in sorted(type_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {t}: {stats['correct']}/{stats['total']} ({acc:.1%})")

    # Save results
    if output_file:
        output_data = {
            "summary": {
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "model": model,
                "repeat_problem": repeat_problem,
                "k_shot": k_shot,
            },
            "by_type": {t: {"correct": s["correct"], "total": s["total"],
                          "accuracy": s["correct"]/s["total"] if s["total"] > 0 else 0}
                       for t, s in type_stats.items()},
            "results": results,
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate compositional questions")
    parser.add_argument("--input", "-i", default="compositional_questions.json", help="Input file")
    parser.add_argument("--output", "-o", default=None, help="Output file")
    parser.add_argument("--model", "-m", default="claude-opus-4-5-20251101", help="Model to use")
    parser.add_argument("--repeat", "-r", type=int, default=None, help="Repeat question N times")
    parser.add_argument("--concurrency", "-c", type=int, default=40, help="Concurrent requests")
    parser.add_argument("--k-shot", "-k", type=int, default=10, help="Number of few-shot examples")
    parser.add_argument("--verbosity", "-v", type=int, default=2, help="Verbosity level")
    args = parser.parse_args()

    # Set default output file
    if args.output is None:
        suffix = f"_r{args.repeat}" if args.repeat and args.repeat > 1 else ""
        args.output = f"eval_results/compositional{suffix}.json"

    os.makedirs("eval_results", exist_ok=True)

    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Repeat: {args.repeat}")
    print(f"  K-shot: {args.k_shot}")

    asyncio.run(run_evaluation(
        args.input,
        args.output,
        model=args.model,
        repeat_problem=args.repeat,
        concurrency=args.concurrency,
        k_shot=args.k_shot,
        verbosity=args.verbosity,
    ))


if __name__ == "__main__":
    main()
