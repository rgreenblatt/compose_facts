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
from openai import AsyncOpenAI
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

# Load API keys
if "ANTHROPIC_API_KEY" not in os.environ:
    key_path = os.path.expanduser("~/.anthropic_api_key")
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
    filler_tokens=None,
):
    """Evaluate a single question."""
    max_tokens = 50

    # Determine model type
    is_openai_chat = model in OPENAI_CHAT_MODELS
    is_gemini = model in GEMINI_MODELS
    is_openrouter = model in OPENROUTER_MODELS or is_gemini

    messages = build_full_prompt(
        questions=questions,
        question_index=question_index,
        default_few_shot_indices=few_shot_indices,
        repeat_problem=repeat_problem,
        add_cache_control=not is_openai_chat and not is_openrouter,
        filler_tokens=filler_tokens,
    )
    question = questions[question_index]

    # Build cache_key based on model type
    if is_openai_chat or is_openrouter:
        # Convert messages to OpenAI format (simple string content)
        openai_messages = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                content = content[0]["text"]
            openai_messages.append({"role": msg["role"], "content": content})

        if is_gemini:
            cache_key = {
                "model": model,
                "max_completion_tokens": 20,
                "messages": openai_messages,
                "temperature": 0.3,
                "extra_body": {
                    "reasoning": {
                        "enabled": True,
                        "thinking_level": "low",
                        "max_tokens": 10,
                    }
                },
            }
        elif "gpt-5" in model:
            cache_key = {
                "model": model,
                "max_completion_tokens": max_tokens,
                "messages": openai_messages,
                "reasoning_effort": "none",
            }
        else:
            cache_key = {"model": model, "max_tokens": max_tokens, "messages": openai_messages}
    else:
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
                    if is_openrouter:
                        # OpenRouter API (includes Gemini)
                        if is_gemini:
                            # Gemini: retry until non-empty and no thinking
                            gemini_max_retries = 5
                            for gemini_retry in range(gemini_max_retries):
                                response = await asyncio.wait_for(
                                    openrouter_client.chat.completions.create(**cache_key),
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
                                    if verbosity >= 2:
                                        print(f"  Gemini model {model} returned reasoning, retrying ({gemini_retry + 1}/{gemini_max_retries}). Thinking: {thinking[:100]}")
                                elif response_text == "":
                                    if verbosity >= 2:
                                        print(f"  Gemini returned empty response, retrying ({gemini_retry + 1}/{gemini_max_retries})...")
                                else:
                                    assert thinking is None and response_text != ""
                                    if verbosity >= 2 and gemini_retry > 0:
                                        print(f"  Gemini succeeded on retry {gemini_retry + 1}")
                                    break
                            else:
                                if verbosity >= 2:
                                    print(f"  Gemini returned empty/thinking response after {gemini_max_retries} retries")
                        else:
                            response = await asyncio.wait_for(
                                openrouter_client.chat.completions.create(
                                    **cache_key,
                                    temperature=0.0,
                                ),
                                timeout=120.0,
                            )
                            response_text = response.choices[0].message.content.strip()
                    elif is_openai_chat:
                        # OpenAI chat API
                        response = await asyncio.wait_for(
                            openai_client.chat.completions.create(
                                **cache_key,
                                temperature=0.0,
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
                                temperature=0.0,
                            ),
                            timeout=120.0,
                        )
                        response_text = response.content[0].text.strip()

                    # Cache the response
                    await cache.set(cache_key, {"response": response_text})
                    break

                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        if attempt < max_retries - 1:
                            wait_time = 0.1 * (2 ** attempt)
                            print(f"Rate limited on question {question_index + 1}, waiting {wait_time}s ({attempt + 1}/{max_retries})...")
                            await asyncio.sleep(wait_time)
                            continue
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

            assert response_text is not None

        # Parse response based on answer type
        answer_type = question.get("answer_type", "number")
        if answer_type == "element_name":
            # Parse string answer (element name)
            cleaned = (
                response_text.strip()
                .lower()
                .removeprefix("answer:")
                .strip()
            )
            # Extract first word as the element name
            predicted = cleaned.split()[0] if cleaned.split() else cleaned
        else:
            # Parse numerical answer
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
    filler_tokens=None,
):
    """Run evaluation on all questions."""
    # Load questions
    questions = load_questions(input_file)
    print(f"Loaded {len(questions)} questions from {input_file}")

    _, few_shot_indices = select_few_shot_questions(
        questions, k_shot=k_shot, seed=42
    )

    # Initialize cache (model-specific)
    cache_suffix = f"_{model.replace('-', '_').replace('/', '_')}"
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
            model=model, repeat_problem=repeat_problem, verbosity=verbosity,
            filler_tokens=filler_tokens
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

    # Print Gemini failures if applicable
    if model in GEMINI_MODELS:
        gemini_thinking_count = sum(1 for r in results if r.get("response") == "INVALID WAS THINKING")
        gemini_empty_count = sum(1 for r in results if r.get("response") == "")
        if gemini_thinking_count > 0 or gemini_empty_count > 0:
            print(f"Gemini failures: {gemini_thinking_count} thinking, {gemini_empty_count} empty")

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
                "filler_tokens": filler_tokens,
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
    parser = argparse.ArgumentParser(description="Evaluate compositional questions")
    parser.add_argument("--input", "-i", default="compositional_questions.json", help="Input file")
    parser.add_argument("--output", "-o", default=None, help="Output file")
    parser.add_argument("--model", "-m", default="opus-4-5", help="Model to use")
    parser.add_argument("--repeat", "-r", type=int, default=None, help="Repeat question N times")
    parser.add_argument("--concurrency", "-c", type=int, default=40, help="Concurrent requests")
    parser.add_argument("--k-shot", "-k", type=int, default=10, help="Number of few-shot examples")
    parser.add_argument("--verbosity", "-v", type=int, default=2, help="Verbosity level")
    parser.add_argument("--filler-tokens", "-f", type=int, default=None,
                        help="Number of filler tokens (counting 1 to N) to add after the problem")
    args = parser.parse_args()

    model = parse_model_name(args.model)

    # Set default output file
    if args.output is None:
        repeat_suffix = f"_r{args.repeat}" if args.repeat and args.repeat > 1 else ""
        filler_suffix = f"_f{args.filler_tokens}" if args.filler_tokens else ""
        args.output = f"eval_results/compositional_{args.model}{repeat_suffix}{filler_suffix}.json"

    os.makedirs("eval_results", exist_ok=True)

    print(f"Configuration:")
    print(f"  Model: {model}")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Repeat: {args.repeat}")
    print(f"  K-shot: {args.k_shot}")
    if args.filler_tokens:
        print(f"  Filler tokens: {args.filler_tokens}")

    asyncio.run(run_evaluation(
        args.input,
        args.output,
        model=model,
        repeat_problem=args.repeat,
        concurrency=args.concurrency,
        k_shot=args.k_shot,
        verbosity=args.verbosity,
        filler_tokens=args.filler_tokens,
    ))


if __name__ == "__main__":
    main()
