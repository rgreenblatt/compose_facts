from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
import json
import asyncio
import argparse
import os
from prompt_builder import build_few_shot_messages, build_user_message, load_questions, select_few_shot_questions

# Load API keys
if "ANTHROPIC_API_KEY" not in os.environ:
    key_path = os.path.expanduser("~/.anthropic_api_key_rr")
    try:
        with open(key_path, "r") as f:
            os.environ["ANTHROPIC_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        pass

if "OPENAI_API_KEY" not in os.environ:
    key_path = os.path.expanduser("~/.openai_api_key")
    try:
        with open(key_path, "r") as f:
            os.environ["OPENAI_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        pass

if "OPENROUTER_API_KEY" not in os.environ:
    key_path = os.path.expanduser("~/.openrouter_api_key")
    try:
        with open(key_path, "r") as f:
            os.environ["OPENROUTER_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        pass

# Initialize clients
client = AsyncAnthropic()
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
openrouter_client = AsyncOpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# OpenAI models (chat API)
OPENAI_CHAT_MODELS = {
    "gpt-3.5-turbo-0125",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-0125-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-1106-preview",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "gpt-4.1-2025-04-14",
    "gpt-5.1-2025-11-13",
    "gpt-5.2-2025-12-11",
}

# OpenRouter models
OPENROUTER_MODELS = {
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-v3.2",
    "qwen/qwen3-235b-a22b",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-coder",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2",
    "google/gemini-2.5-pro",
    "google/gemini-3-pro-preview",
}

# Gemini models (require special handling - retry for thinking/empty responses)
GEMINI_MODELS = {
    "google/gemini-2.5-pro",
    "google/gemini-3-pro-preview",
}

async def run_sample(
    messages,
    semaphore,
    model,
    max_retries=8,
):
    max_tokens = 50

    # Determine model type
    is_openai_chat = model in OPENAI_CHAT_MODELS
    is_gemini = model in GEMINI_MODELS
    is_openrouter = model in OPENROUTER_MODELS or is_gemini

    # Convert messages to OpenAI format if needed
    if is_openai_chat or is_openrouter:
        openai_messages = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                content = content[0]["text"]
            openai_messages.append({"role": msg["role"], "content": content})

    async with semaphore:
        response_text = None
        # Make API call with retry logic
        for attempt in range(max_retries):
            try:
                if is_openrouter:
                    # OpenRouter API (includes Gemini)
                    if is_gemini:
                        # Gemini: retry until non-empty and no thinking
                        gemini_max_retries = 15
                        for gemini_retry in range(gemini_max_retries):
                            response = await asyncio.wait_for(
                                openrouter_client.chat.completions.create(
                                    model=model,
                                    max_completion_tokens=20,
                                    messages=openai_messages,
                                    temperature=1.0,
                                    extra_body={
                                        "reasoning": {
                                            "enabled": True,
                                            "thinking_level": "low",
                                            "max_tokens": 10,
                                        }
                                    },
                                ),
                                timeout=120.0,
                            )
                            assert response.choices is not None
                            choice = response.choices[0]
                            assert choice is not None
                            response_text = choice.message.content.strip()
                            thinking = None
                            if hasattr(choice.message, "reasoning") and choice.message.reasoning:
                                thinking = choice.message.reasoning
                            elif hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
                                thinking = choice.message.reasoning_content

                            if thinking is not None:
                                response_text = "INVALID WAS THINKING"
                                print(f"  Gemini model {model} returned reasoning, retrying ({gemini_retry + 1}/{gemini_max_retries})")
                            elif response_text == "":
                                print(f"  Gemini returned empty response, retrying ({gemini_retry + 1}/{gemini_max_retries})...")
                            else:
                                break
                        else:
                            print(f"  Gemini returned empty/thinking response after {gemini_max_retries} retries")
                    else:
                        response = await asyncio.wait_for(
                            openrouter_client.chat.completions.create(
                                model=model,
                                max_tokens=max_tokens,
                                messages=openai_messages,
                                temperature=1.0,
                            ),
                            timeout=120.0,
                        )
                        response_text = response.choices[0].message.content.strip()
                elif is_openai_chat:
                    # OpenAI chat API
                    response = await asyncio.wait_for(
                        openai_client.chat.completions.create(
                            model=model,
                            max_tokens=max_tokens,
                            messages=openai_messages,
                            temperature=1.0,
                        ),
                        timeout=120.0,
                    )
                    response_text = response.choices[0].message.content.strip()
                else:
                    # Anthropic API
                    response = await asyncio.wait_for(
                        client.messages.create(
                            model=model,
                            max_tokens=max_tokens,
                            messages=messages,
                            temperature=1.0,
                        ),
                        timeout=120.0,
                    )
                    response_text = response.content[0].text.strip()

                break

            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = 0.1 * (2 ** attempt)
                        print(f"Rate limited for model {model}, waiting {wait_time}s ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                        continue
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt * 0.25
                    print(f"Error for model {model} ({attempt=}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"Failed after {max_retries} attempts: {e}")
                    return None

        assert response_text is not None
    return response_text

def parse_response(response_text, is_int: bool):
    """Parse response to extract numeric/str answer."""
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
        if "(" in cleaned:
            print(f"  Removing parenthetical from response: {cleaned}")
            cleaned = cleaned.split("(")[0].strip()
        if is_int:
            final = cleaned.split()[0] if cleaned.split() else cleaned
            return int(final)
        return cleaned
    except (ValueError, IndexError):
        return None


async def async_main(messages, n, correct_answer, models=None):
    DEFAULT_MODELS = [
        # ("claude-3-5-haiku-20241022", "Haiku 3.5"),
        # ("claude-sonnet-4-20250514", "Sonnet 4"),
        ("claude-opus-4-20250514", "Opus 4"),
        ("claude-opus-4-5-20251101", "Opus 4.5"),
        ("google/gemini-3-pro-preview", "Gemini 3 Pro"),
    ]
    MODELS = models if models else DEFAULT_MODELS
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
        parsed = [parse_response(r, is_int=isinstance(correct_answer, int)) for r in responses]
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



def parse_model_name(model_shorthand):
    """Convert model shorthand to full model ID."""
    model_map = {
        # Anthropic models
        "opus-4-5": "claude-opus-4-5-20251101",
        "opus-4": "claude-opus-4-20250514",
        "sonnet-4": "claude-sonnet-4-20250514",
        "sonnet-4-5": "claude-sonnet-4-5-20250929",
        "haiku-3-5": "claude-3-5-haiku-20241022",
        # OpenAI models
        "gpt-3.5": "gpt-3.5-turbo-0125",
        "gpt-4": "gpt-4-0314",
        "gpt-4o": "gpt-4o-2024-05-13",
        "gpt-4.1": "gpt-4.1-2025-04-14",
        "gpt-5.1": "gpt-5.1-2025-11-13",
        "gpt-5.2": "gpt-5.2-2025-12-11",
        # OpenRouter models
        "deepseek-v3": "deepseek/deepseek-chat-v3-0324",
        "deepseek-v3.2": "deepseek/deepseek-v3.2",
        "qwen3-235b": "qwen/qwen3-235b-a22b-2507",
        "qwen3-480b": "qwen/qwen3-coder",
        "qwen3-32b": "qwen/qwen3-32b",
        "kimi-k2": "moonshotai/kimi-k2",
        # Gemini models
        "gemini-2.5-pro": "google/gemini-2.5-pro",
        "gemini-3-pro": "google/gemini-3-pro-preview",
    }
    return model_map.get(model_shorthand, model_shorthand)


def main():
    parser = argparse.ArgumentParser(description="Show random prompts for compositional questions")
    parser.add_argument("--input", "-i", default="compositional_questions.json", help="Input file")
    parser.add_argument("--count", "-n", type=int, default=64, help="Number of samples")
    parser.add_argument("--k-shot", "-k", type=int, default=10, help="Number of few-shot examples")
    parser.add_argument("--repeat", "-r", type=int, default=1, help="Repeat question N times")
    parser.add_argument("--exclude-atomic", action="store_true", help="Exclude questions containing 'atomic' from few-shot prompt")
    parser.add_argument("--harder", action="store_true", help="Do harder version of the question")
    parser.add_argument("--filler-tokens", "-f", type=int, default=None,
                        help="Number of filler tokens (counting 1 to N) to add after the problem")
    parser.add_argument("--models", "-m", type=str, nargs="+", default=None,
                        help="Models to evaluate (e.g., opus-4-5 gemini-2.5-pro)")
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
        add_cache_control=False,
        filler_tokens=args.filler_tokens
    )

    # Assert no answer leakage in few-shot prompt
    few_shot_str = str(few_shot_messages).lower()

    # The original Manifold market question
    if args.harder:
        assert "cube" not in few_shot_str, "Few-shot prompt contains 'cube' - answer leakage!"
        assert "euler" not in few_shot_str, "Few-shot prompt contains 'euler' - answer leakage!"
        question_text  = "What is the name of the element with an atomic number equal to the sum of the age at which Euler died and the number of faces on a cube?"
        correct_answer = None; assert correct_answer is not None, 'fill in correct answer with the right answer before running' # DO NOT COMMIT THE CORRECT ANSWER TO AVOID LEAKAGE!!!
        assert correct_answer not in few_shot_str, "Few-shot prompt has answer leakage!"
    else:
        assert "uranium" not in few_shot_str, "Few-shot prompt contains 'uranium' - answer leakage!"
        assert "euler" not in few_shot_str, "Few-shot prompt contains 'euler' - answer leakage!"
        question_text = "What is the sum of the atomic number of uranium and the age at which Euler died?"
        correct_answer = None; assert correct_answer is not None, 'fill in correct answer with the right answer before running' # DO NOT COMMIT THE CORRECT ANSWER TO AVOID LEAKAGE!!!

    # Build messages
    messages = few_shot_messages + [
        {
            "role": "user",
            "content": build_user_message(
                {"question": question_text},
                repeat_problem=args.repeat,
                non_numerical=args.harder,
                filler_tokens=args.filler_tokens
            ),
        },
        {"role": "assistant", "content": "Answer:"},  # prefill
    ]

    print(f"Question: {question_text}")
    if args.filler_tokens:
        print(f"Filler tokens: {args.filler_tokens}")

    print("Full prompt:")
    print(json.dumps(messages, indent=2))

    # Parse model names if provided
    models = None
    if args.models:
        models = [(parse_model_name(m), m) for m in args.models]
        print(f"\nUsing models: {[m[0] for m in models]}")

    asyncio.run(async_main(messages, n=args.count, correct_answer=correct_answer, models=models))


if __name__ == "__main__":
    main()
