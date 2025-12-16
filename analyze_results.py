#!/usr/bin/env python3
"""
Analyze and compare evaluation results.
"""

import json


def load_results(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def analyze_errors(results):
    """Analyze what questions the model got wrong."""
    errors = [r for r in results["results"] if not r["is_correct"]]
    return errors


def main():
    # Load results
    r1 = load_results("eval_results/opus_4_5_r1.json")
    r5 = load_results("eval_results/opus_4_5_r5.json")

    print("=" * 70)
    print("COMPOSITIONAL REASONING EVALUATION RESULTS")
    print("=" * 70)

    print("\n## Summary")
    print(f"\n{'Metric':<30} {'r=1':<15} {'r=5':<15} {'Δ':<10}")
    print("-" * 70)
    print(f"{'Overall Accuracy':<30} {r1['summary']['accuracy']*100:.1f}%{'':<10} {r5['summary']['accuracy']*100:.1f}%{'':<10} +{(r5['summary']['accuracy']-r1['summary']['accuracy'])*100:.1f}%")

    print("\n## By Composition Type")
    print(f"\n{'Type':<20} {'r=1':<15} {'r=5':<15} {'Δ':<10}")
    print("-" * 70)

    for comp_type in sorted(r1["by_type"].keys()):
        acc1 = r1["by_type"][comp_type]["accuracy"] * 100
        acc5 = r5["by_type"][comp_type]["accuracy"] * 100
        delta = acc5 - acc1
        print(f"{comp_type:<20} {acc1:.1f}%{'':<10} {acc5:.1f}%{'':<10} {delta:+.1f}%")

    # Error analysis for r=1
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS (r=1)")
    print("=" * 70)

    errors_r1 = analyze_errors(r1)
    print(f"\nTotal errors: {len(errors_r1)}")

    # Group by composition type
    by_type = {}
    for e in errors_r1:
        t = e.get("composition_type", "unknown")
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(e)

    for comp_type, errors in sorted(by_type.items()):
        print(f"\n### {comp_type} ({len(errors)} errors)")
        for e in errors[:3]:  # Show first 3 of each type
            print(f"  Q: {e['question'][:80]}...")
            print(f"     Expected: {e['correct_answer']}, Got: {e['predicted_answer']}")
            if e['predicted_answer'] is not None:
                diff = e['predicted_answer'] - e['correct_answer']
                print(f"     Diff: {diff:+d}")
            print()

    # Check if r=5 fixed any errors
    print("=" * 70)
    print("QUESTIONS FIXED BY REPETITION")
    print("=" * 70)

    r1_errors = {r["question_index"] for r in r1["results"] if not r["is_correct"]}
    r5_errors = {r["question_index"] for r in r5["results"] if not r["is_correct"]}

    fixed = r1_errors - r5_errors
    broken = r5_errors - r1_errors

    print(f"\nFixed by r=5: {len(fixed)}")
    print(f"Broken by r=5: {len(broken)}")
    print(f"Net improvement: {len(fixed) - len(broken)}")

    # Show examples of fixed questions
    if fixed:
        print("\nExamples of questions fixed by repetition:")
        r1_by_idx = {r["question_index"]: r for r in r1["results"]}
        r5_by_idx = {r["question_index"]: r for r in r5["results"]}
        for idx in list(fixed)[:5]:
            r1_q = r1_by_idx[idx]
            r5_q = r5_by_idx[idx]
            print(f"  Q: {r1_q['question'][:70]}...")
            print(f"     r=1 answer: {r1_q['predicted_answer']}, r=5 answer: {r5_q['predicted_answer']}, correct: {r1_q['correct_answer']}")
            print()


if __name__ == "__main__":
    main()
