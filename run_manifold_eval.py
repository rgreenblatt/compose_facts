from anthropic import AsyncAnthropic
import json
import asyncio
import argparse
import os
from prompt_builder import build_few_shot_messages, build_user_message, load_questions, select_few_shot_questions

# Load API key
if "ANTHROPIC_API_KEY" not in os.environ:
    key_path = os.path.expanduser("~/.anthropic_api_key_rr")
    try:
        with open(key_path, "r") as f:
            os.environ["ANTHROPIC_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        pass

client = AsyncAnthropic()

async def run_sample(
    messages,
    semaphore,
    model,
    max_retries=8,

):
    max_tokens = 50
    async with semaphore:
        response_text = None
        # Make API call with retry logic
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        messages=messages,
                        temperature=1.0,
                    ),
                    timeout=120.0,
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt * 0.25
                    print(f"Error for model {model} ({attempt=}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"Failed after {max_retries} attempts: {e}")
                    return None

            response_text = response.content[0].text.strip()

            break

        assert response_text is not None
    return response_text

def parse_response(response_text):
    """Parse response to extract numeric answer."""
    if response_text is None:
        return None
    try:
        cleaned = (
            response_text.strip()
            .lower()
            .replace(",", "")
            .removeprefix("answer:")
            .strip()
        )
        return int(cleaned.split()[0] if cleaned.split() else cleaned)
    except (ValueError, IndexError):
        return None


async def async_main(messages, n, correct_answer):
    MODELS = [
        ("claude-3-5-haiku-20241022", "Haiku 3.5"),
        ("claude-sonnet-4-20250514", "Sonnet 4"),
        ("claude-opus-4-20250514", "Opus 4"),
        ("claude-opus-4-5-20251101", "Opus 4.5"),
    ]
    semaphore = asyncio.Semaphore(20)  # Limit concurrent requests

    print(f"\nRunning {n} samples per model at temperature=1.0")
    print(f"Correct answer: {correct_answer}")
    print("=" * 60)

    results = {}

    for model_id, model_name in MODELS:
        print(f"\n{model_name} ({model_id})...")

        # Run n samples concurrently
        tasks = [run_sample(messages, semaphore, model_id) for _ in range(n)]
        responses = await asyncio.gather(*tasks)

        # Parse and score
        parsed = [parse_response(r) for r in responses]
        correct = sum(1 for p in parsed if p == correct_answer)
        accuracy = correct / n

        # Count unique answers
        answer_counts = {}
        for p in parsed:
            answer_counts[p] = answer_counts.get(p, 0) + 1

        results[model_name] = {
            "correct": correct,
            "total": n,
            "accuracy": accuracy,
            "answer_distribution": answer_counts,
        }

        print(f"  Accuracy: {correct}/{n} ({accuracy:.1%})")
        print(f"  Top answers: {sorted(answer_counts.items(), key=lambda x: -x[1])[:5]}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'Accuracy':<15} {'Correct':<10}")
    print("-" * 40)
    for model_name in [m[1] for m in MODELS]:
        r = results[model_name]
        print(f"{model_name:<15} {r['accuracy']:.1%}{'':<10} {r['correct']}/{r['total']}")

    return results



def main():
    parser = argparse.ArgumentParser(description="Show random prompts for compositional questions")
    parser.add_argument("--input", "-i", default="compositional_questions.json", help="Input file")
    parser.add_argument("--count", "-n", type=int, default=64, help="Number of samples")
    parser.add_argument("--k-shot", "-k", type=int, default=10, help="Number of few-shot examples")
    parser.add_argument("--repeat", "-r", type=int, default=1, help="Repeat question N times")
    parser.add_argument("--exclude-atomic", action="store_true", help="Exclude questions containing 'atomic' from few-shot prompt")
    args = parser.parse_args()

    questions = load_questions(args.input)
    print(f"Loaded {len(questions)} questions from {args.input}")

    # Filter out atomic questions if requested (before selection)
    candidate_questions = questions
    if args.exclude_atomic:
        candidate_questions = [
            q for q in questions
            if "atomic" not in q["question"].lower() and
               "atomic" not in q.get("component1", {}).get("description", "").lower() and
               "atomic" not in q.get("component2", {}).get("description", "").lower()
        ]
        filtered_count = len(questions) - len(candidate_questions)
        print(f"Filtered out {filtered_count} questions containing 'atomic'")
        print(f"Selecting few-shot from {len(candidate_questions)} remaining questions")

    # Select few-shot examples from candidate pool
    few_shot_questions, _ = select_few_shot_questions(
        candidate_questions, k_shot=args.k_shot, seed=42
    )

    few_shot_messages = build_few_shot_messages(
        few_shot_questions,
        repeat_problem=args.repeat,
        add_cache_control=False
    )

    # Assert no answer leakage in few-shot prompt
    few_shot_str = str(few_shot_messages).lower()
    assert "uranium" not in few_shot_str, "Few-shot prompt contains 'uranium' - answer leakage!"
    assert "euler" not in few_shot_str, "Few-shot prompt contains 'euler' - answer leakage!"

    # The original Manifold market question
    question_text = "What is the sum of the atomic number of uranium and the age at which Euler died?"
    correct_answer = None; assert correct_answer is not None, 'fill in correct answer with the right answer before running' # DO NOT COMMIT THE CORRECT ANSWER TO AVOID LEAKAGE!!!

    # Build messages
    messages = few_shot_messages + [
        {
            "role": "user",
            "content": build_user_message({"question": question_text}, repeat_problem=args.repeat),
        },
        {"role": "assistant", "content": "Answer:"},  # prefill
    ]

    print(f"Question: {question_text}")

    print("Full prompt:")
    print(json.dumps(messages, indent=2))

    asyncio.run(async_main(messages, n=args.count, correct_answer=correct_answer))


if __name__ == "__main__":
    main()
