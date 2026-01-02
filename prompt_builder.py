#!/usr/bin/env python3
"""
Prompt building utilities for compositional reasoning evaluation.
Extracted for reuse and testing.
"""

import json
import random


def load_questions(filepath):
    """Load questions from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def get_component_facts(question):
    """Extract component fact descriptions from a question."""
    return [question["component1"]["description"], question["component2"]["description"]]


def select_few_shot_questions(questions, k_shot=10, seed=42):
    """
    Select k_shot questions for few-shot prompting.

    Args:
        questions: List of all questions
        k_shot: Number of few-shot examples
        seed: Random seed for reproducibility
        minimize_overlap: If True, select questions with the most isolated facts
                         (facts that appear in fewest other questions)

    Returns:
        Tuple of (few_shot_questions, few_shot_indices)
    """
    rng = random.Random(seed)
    indices = list(range(len(questions)))

    rng.shuffle(indices)
    few_shot_indices = indices[:k_shot]
    few_shot_questions = [questions[i] for i in indices[:k_shot]]
    return few_shot_questions, few_shot_indices


def select_few_shot_question_for_problem(questions, question_index, few_shot_indices):
    facts_used_in_q = get_component_facts(questions[question_index])

    overlapping_indices = []
    non_overlapping_indices = []

    for i, q in enumerate(questions):
        q_facts = get_component_facts(q)
        if len(set(q_facts).intersection(set(facts_used_in_q))) == 0:
            non_overlapping_indices.append(i)
        else:
            # print("Overlapping fact found between question", questions[question_index], "and", q, (question_index, i))
            overlapping_indices.append(i)

    assert question_index not in non_overlapping_indices, "should overlap with itself"

    assert len(non_overlapping_indices) > len(
        few_shot_indices
    ), "Not enough non-overlapping questions to select few-shot examples."

    new_few_shot_indices = []
    for idx in few_shot_indices:
        if idx in overlapping_indices or idx in new_few_shot_indices:
            # get nearest non-overlapping index
            replacement_idx = min(non_overlapping_indices, key=lambda x: abs(x - idx))
        else:
            replacement_idx = idx
            assert replacement_idx in non_overlapping_indices, "should be non-overlapping"

        new_few_shot_indices.append(replacement_idx)
        non_overlapping_indices.remove(replacement_idx)

    new_few_shot_questions = [questions[i] for i in new_few_shot_indices]

    return new_few_shot_questions, new_few_shot_indices


def build_user_message(question, repeat_problem=None, non_numerical: bool | None = None, filler_tokens: int | None = None, long_prefill: bool = False):
    """Build the user message for a question.

    Args:
        question: The question dict or string
        repeat_problem: Number of times to repeat the problem (None = 1 time)
        non_numerical: Whether the answer is non-numerical (auto-detected if None)
        filler_tokens: Number of filler tokens (counting 1 to N) to add after the problem
        long_prefill: If True, use longer prefill format
    """
    # Auto-detect non_numerical from question if not explicitly set
    if isinstance(question, dict) and non_numerical is None:
        non_numerical = question.get("answer_type") == "element_name"

    if long_prefill:
        instruction = (
            "You will be given a question that requires combining two facts and adding them"
            + (" and processing this to get a final answer" if non_numerical else "")
            + ". "
            + "Answer immediately using the format 'I will now answer immediately with the answer. The answer is [ANSWER]' where [ANSWER] is just the "
            + ("" if non_numerical else "numerical ")
            + "answer, nothing else. No explanation, no words, no reasoning, just the "
            + ("answer." if non_numerical else "number.")
        )
    else:
        instruction = (
            "You will be given a question that requires combining two facts and adding them"
            + (" and processing this to get a final answer" if non_numerical else "")
            + ". "
            + "Answer immediately using the format 'Answer: [ANSWER]' where [ANSWER] is just the "
            + ("" if non_numerical else "numerical ")
            + "answer, nothing else. No explanation, no words, no reasoning, just the "
            + ("answer." if non_numerical else "number.")
        )

    if filler_tokens is not None:
        instruction += f" After the question, there will be filler tokens (counting from 1 to {filler_tokens}) to give you extra space to process the problem before answering."

    q_text = question["question"] if isinstance(question, dict) else question

    if repeat_problem is None or repeat_problem == 1:
        out = f"{instruction}\n\nQuestion: {q_text}"
    else:
        parts = [f"Question: {q_text}"]
        for i in range(1, repeat_problem):
            parts.append(f"Question (repeat #{i+1}): {q_text}")
        out = f"{instruction}\n\n" + "\n\n".join(parts)

    # Add filler tokens if specified
    if filler_tokens is not None:
        filler = " ".join(str(i) for i in range(1, filler_tokens + 1))
        out += f"\n\nFiller: {filler}"

    return out


def build_few_shot_messages(few_shot_questions, repeat_problem=None, add_cache_control=True, filler_tokens=None, long_prefill=False):
    """Build the few-shot messages as user/assistant pairs."""
    messages = []

    for question in few_shot_questions:
        user_text = build_user_message(question, repeat_problem=repeat_problem, filler_tokens=filler_tokens, long_prefill=long_prefill)

        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            }
        )
        if long_prefill:
            assistant_content = f"I will now answer immediately with the answer. The answer is {question['answer']}"
        else:
            assistant_content = f"Answer: {question['answer']}"
        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
            }
        )

    # Add cache control to last user message
    if add_cache_control and messages:
        messages[-2]["content"][0]["cache_control"] = {"type": "ephemeral"}

    return messages


def build_full_prompt(
    questions, question_index, default_few_shot_indices, repeat_problem=None, add_cache_control=True, verbosity=1, filler_tokens=None, long_prefill=False
):

    few_shot_questions, new_few_shot_indices = select_few_shot_question_for_problem(
        questions, question_index, default_few_shot_indices
    )
    assert len(few_shot_questions) == len(default_few_shot_indices)

    using_default_few_shot_indices = new_few_shot_indices == default_few_shot_indices
    if verbosity >= 2 and not using_default_few_shot_indices:
        print(
            f"Replaced few-shot indices {default_few_shot_indices} with {new_few_shot_indices} for question index {question_index} to avoid fact overlap."
        )

    question = questions[question_index]

    for fsq in few_shot_questions:
        assert question["question"] != fsq["question"], (
            "Test question should not be in few-shot examples.",
            question["question"],
            fsq["question"],
        )
        assert (
            len(set(get_component_facts(question)).intersection(set(get_component_facts(fsq)))) == 0
        ), "Few-shot questions should not share facts with the test question."

    few_shot_messages = build_few_shot_messages(
        few_shot_questions,
        repeat_problem=repeat_problem,
        add_cache_control=add_cache_control and using_default_few_shot_indices,
        filler_tokens=filler_tokens,
        long_prefill=long_prefill,
    )

    # Build messages
    if long_prefill:
        prefill = "I will now answer immediately with the answer. The answer is"
    else:
        prefill = "Answer:"
    messages = few_shot_messages + [
        {
            "role": "user",
            "content": build_user_message(question, repeat_problem=repeat_problem, filler_tokens=filler_tokens, long_prefill=long_prefill),
        },
        {"role": "assistant", "content": prefill},  # prefill
    ]

    return messages
