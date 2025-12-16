#!/usr/bin/env python3
"""
Run all element name compositional reasoning experiments across models and repetition settings.
"""

import subprocess
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


MODELS = [
    ("claude-3-5-haiku-20241022", "Haiku 3.5"),
    ("claude-sonnet-4-20250514", "Sonnet 4"),
    ("claude-opus-4-20250514", "Opus 4"),
    ("claude-opus-4-5-20251101", "Opus 4.5"),
]

REPEAT_VALUES = [1, 5]

# Models to test with k_shot=0
ZERO_SHOT_MODELS = [
    ("claude-sonnet-4-20250514", "Sonnet 4"),
    ("claude-opus-4-20250514", "Opus 4"),
    ("claude-opus-4-5-20251101", "Opus 4.5"),
]


def run_evaluation(model_id, model_name, repeat, k_shot=10):
    """Run a single evaluation."""
    # Create model-specific output filename
    model_short = model_name.lower().replace(" ", "_").replace(".", "_")
    output_file = f"eval_results/element_{model_short}_k{k_shot}_r{repeat}.json"

    cmd = [
        "python", "eval_compositional.py",
        "--input", "compositional_element_questions.json",
        "--model", model_id,
        "--repeat", str(repeat),
        "--k-shot", str(k_shot),
        "--output", output_file,
        "--verbosity", "1",  # Less verbose
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return output_file


def load_results(filepath):
    """Load results from JSON file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        return json.load(f)


def load_problem_results(model_name, repeat, k_shot=10):
    """Load individual problem results from output file."""
    model_short = model_name.lower().replace(" ", "_").replace(".", "_")
    output_file = f"eval_results/element_{model_short}_k{k_shot}_r{repeat}.json"

    if not os.path.exists(output_file):
        return None, None

    with open(output_file, "r") as f:
        data = json.load(f)

    # Return dict of {problem_index: is_correct (0 or 1)}
    results = {}
    problems = {}
    for result in data["results"]:
        problem_idx = result["question_index"]
        results[problem_idx] = 1 if result["is_correct"] else 0
        problems[problem_idx] = result["question"]

    return results, problems


def paired_ttest(model_name, r1, r2, k_shot=10):
    """
    Perform paired t-test on problem-level results between two repetition values.

    Returns:
        (mean_diff, p_value) where mean_diff is r2 - r1 in percentage points
    """
    # Load results for both repetition values
    results_r1, problems_r1 = load_problem_results(model_name, r1, k_shot)
    results_r2, problems_r2 = load_problem_results(model_name, r2, k_shot)

    if results_r1 is None or results_r2 is None:
        return 0.0, 1.0

    # Verify same problems were tested
    assert set(results_r1.keys()) == set(results_r2.keys()), "Different problems tested"

    # Get common problem indices
    common_indices = sorted(results_r1.keys())

    if len(common_indices) == 0:
        return 0.0, 1.0

    # Create paired arrays
    scores_r1 = np.array([results_r1[idx] for idx in common_indices])
    scores_r2 = np.array([results_r2[idx] for idx in common_indices])

    # Calculate mean difference in percentage points
    mean_diff = (scores_r2.mean() - scores_r1.mean()) * 100

    # Perform paired t-test
    if np.array_equal(scores_r1, scores_r2):
        # Identical results
        p_value = 1.0
    else:
        t_stat, p_value = stats.ttest_rel(scores_r2, scores_r1)

    return mean_diff, p_value


def plot_bar_chart(all_results):
    """
    Create a grouped bar chart showing r=1 and r=5 accuracy for each model.
    Includes 95% CI error bars and p-values from paired t-test.
    """
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract data
    models = [m[1] for m in MODELS]
    n_models = len(models)
    x = np.arange(n_models)
    bar_width = 0.35

    # Calculate 95% CI for each bar
    def calc_ci(acc, n):
        """Calculate 95% confidence interval using normal approximation."""
        if acc == 0 or acc == 1:
            # Use Wilson score interval for edge cases
            return 1.96 * np.sqrt((acc * (1 - acc) + 1 / (4 * n)) / n) * 100
        return 1.96 * np.sqrt(acc * (1 - acc) / n) * 100

    # Extract accuracies and calculate CIs
    r1_accs = []
    r1_cis = []
    r5_accs = []
    r5_cis = []
    p_values = []

    for model_name in models:
        r1_data = all_results.get(model_name, {}).get("k10_r1", {})
        r5_data = all_results.get(model_name, {}).get("k10_r5", {})

        r1_acc = r1_data.get("accuracy", 0)
        r5_acc = r5_data.get("accuracy", 0)
        r1_n = r1_data.get("total", 100)
        r5_n = r5_data.get("total", 100)

        r1_accs.append(r1_acc * 100)
        r5_accs.append(r5_acc * 100)
        r1_cis.append(calc_ci(r1_acc, r1_n))
        r5_cis.append(calc_ci(r5_acc, r5_n))

        # Calculate p-value
        _, p_val = paired_ttest(model_name, 1, 5, k_shot=10)
        p_values.append(p_val)

    # Plot bars
    bars_r1 = ax.bar(x - bar_width/2, r1_accs, bar_width, yerr=r1_cis,
                     label="r=1", color="#4A90A4", capsize=5,
                     error_kw={"elinewidth": 1.5, "capthick": 1.5})

    bars_r5 = ax.bar(x + bar_width/2, r5_accs, bar_width, yerr=r5_cis,
                     label="r=5", color="#E07B53", capsize=5,
                     error_kw={"elinewidth": 1.5, "capthick": 1.5})

    # Format p-value for display
    def format_p(p):
        if p < 0.001:
            return "p<.001"
        elif p < 0.01:
            return f"p={p:.3f}"
        else:
            return f"p={p:.2f}"

    # Add p-values above r=5 bars
    for i in range(n_models):
        p_text = format_p(p_values[i])
        bar_height = r5_accs[i] + r5_cis[i]
        ax.text(x[i] + bar_width/2, bar_height + 1, p_text,
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Customize plot
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Element Name Compositional Reasoning: Varying Models and Repetition\n(95% CI, paired t-test p-values vs r=1 baseline)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plot_filename = "eval_results/element_bar_chart.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"\nBar chart saved to: {plot_filename}")
    plt.close()


def plot_zero_shot_comparison(all_results):
    """
    Create a bar chart comparing k_shot=0 vs k_shot=10 with r=1 and r=5.
    Shows all 4 combinations for Opus 4.5 and Sonnet 4.
    P-values compare to baseline (k_shot=0, r=1).
    """
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract data for zero-shot models only
    models = [m[1] for m in ZERO_SHOT_MODELS]
    n_models = len(models)
    x = np.arange(n_models)
    bar_width = 0.2

    # Calculate 95% CI for each bar
    def calc_ci(acc, n):
        """Calculate 95% confidence interval using normal approximation."""
        if acc == 0 or acc == 1:
            return 1.96 * np.sqrt((acc * (1 - acc) + 1 / (4 * n)) / n) * 100
        return 1.96 * np.sqrt(acc * (1 - acc) / n) * 100

    # Format p-value for display
    def format_p(p):
        if p < 0.001:
            return "p<.001"
        elif p < 0.01:
            return f"p={p:.3f}"
        else:
            return f"p={p:.2f}"

    # Collect data for all 4 combinations
    combinations = [
        ("k0_r1", 0, 1, "#7FB069", "k=0, r=1"),
        ("k0_r5", 0, 5, "#9BC184", "k=0, r=5"),
        ("k10_r1", 10, 1, "#4A90A4", "k=10, r=1"),
        ("k10_r5", 10, 5, "#E07B53", "k=10, r=5"),
    ]

    bar_data = []
    for key, k_shot, repeat, color, label in combinations:
        accs = []
        cis = []
        p_vals = []

        for model_name in models:
            data = all_results.get(model_name, {}).get(key, {})
            acc = data.get("accuracy", 0)
            n = data.get("total", 100)

            accs.append(acc * 100)
            cis.append(calc_ci(acc, n))

            # Calculate p-value comparing to baseline (k=0, r=1)
            if k_shot == 0 and repeat == 1:
                # This is the baseline
                p_vals.append(None)
            else:
                # Compare to k=0, r=1
                results_baseline, _ = load_problem_results(model_name, 1, k_shot=0)
                results_current, _ = load_problem_results(model_name, repeat, k_shot=k_shot)

                if results_baseline is not None and results_current is not None:
                    common_indices = sorted(set(results_baseline.keys()) & set(results_current.keys()))
                    if len(common_indices) > 0:
                        scores_baseline = np.array([results_baseline[idx] for idx in common_indices])
                        scores_current = np.array([results_current[idx] for idx in common_indices])

                        if np.array_equal(scores_baseline, scores_current):
                            p_vals.append(1.0)
                        else:
                            _, p_val = stats.ttest_rel(scores_current, scores_baseline)
                            p_vals.append(p_val)
                    else:
                        p_vals.append(None)
                else:
                    p_vals.append(None)

        bar_data.append((accs, cis, p_vals, color, label))

    # Plot bars
    for i, (accs, cis, p_vals, color, label) in enumerate(bar_data):
        offset = (i - 1.5) * bar_width
        bars = ax.bar(x + offset, accs, bar_width, yerr=cis,
                      label=label, color=color, capsize=4,
                      error_kw={"elinewidth": 1.5, "capthick": 1.5})

        # Add p-values above bars (except baseline)
        for j in range(n_models):
            if p_vals[j] is not None:
                p_text = format_p(p_vals[j])
                bar_height = accs[j] + cis[j]
                ax.text(x[j] + offset, bar_height + 0.5, p_text,
                        ha="center", va="bottom", fontsize=7, fontweight="bold", rotation=90)

    # Customize plot
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Element Name: Effect of Few-Shot Examples and Repetition\n(95% CI, p-values vs k=0, r=1 baseline)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(loc="upper left", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plot_filename = "eval_results/element_zero_shot_comparison.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"\nZero-shot comparison chart saved to: {plot_filename}")
    plt.close()


def main():
    os.makedirs("eval_results", exist_ok=True)

    print("=" * 70)
    print("ELEMENT NAME COMPOSITIONAL REASONING EXPERIMENT SWEEP")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = {}

    # Standard experiments (k=10) for all models
    for model_id, model_name in MODELS:
        all_results[model_name] = {}

        for repeat in REPEAT_VALUES:
            print(f"\n### {model_name} (k=10, r={repeat})")

            # Generate output filename
            model_short = model_name.lower().replace(" ", "_").replace(".", "_")
            output_file = f"eval_results/element_{model_short}_k10_r{repeat}.json"

            run_evaluation(model_id, model_name, repeat, k_shot=10)
            result = load_results(output_file)
            if result:
                all_results[model_name][f"k10_r{repeat}"] = result["summary"]
            else:
                all_results[model_name][f"k10_r{repeat}"] = {"accuracy": 0, "error": True}

    # Zero-shot experiments (k=0) for selected models
    print("\n" + "=" * 70)
    print("ZERO-SHOT EXPERIMENTS (k=0)")
    print("=" * 70)

    for model_id, model_name in ZERO_SHOT_MODELS:
        for repeat in REPEAT_VALUES:
            print(f"\n### {model_name} (k=0, r={repeat})")

            # Generate output filename
            model_short = model_name.lower().replace(" ", "_").replace(".", "_")
            output_file = f"eval_results/element_{model_short}_k0_r{repeat}.json"

            run_evaluation(model_id, model_name, repeat, k_shot=0)
            result = load_results(output_file)
            if result:
                all_results[model_name][f"k0_r{repeat}"] = result["summary"]
            else:
                all_results[model_name][f"k0_r{repeat}"] = {"accuracy": 0, "error": True}

    # Print summary table for k=10 experiments
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY (k=10)")
    print("=" * 70)

    print("\n| Model | r=1 | r=5 | Δ |")
    print("|-------|-----|-----|---|")

    for model_name in [m[1] for m in MODELS]:
        r1 = all_results.get(model_name, {}).get("k10_r1", {})
        r5 = all_results.get(model_name, {}).get("k10_r5", {})

        acc1 = r1.get("accuracy", 0) * 100
        acc5 = r5.get("accuracy", 0) * 100
        delta = acc5 - acc1 if acc5 > 0 else 0

        print(f"| {model_name} | {acc1:.1f}% | {acc5:.1f}% | {delta:+.1f}% |")

    # Print summary table for zero-shot experiments
    print("\n" + "=" * 70)
    print("ZERO-SHOT COMPARISON (k=0 vs k=10)")
    print("=" * 70)

    print("\n| Model | k=0, r=1 | k=0, r=5 | k=10, r=1 | k=10, r=5 |")
    print("|-------|----------|----------|-----------|-----------|")

    for model_name in [m[1] for m in ZERO_SHOT_MODELS]:
        k0_r1 = all_results.get(model_name, {}).get("k0_r1", {})
        k0_r5 = all_results.get(model_name, {}).get("k0_r5", {})
        k10_r1 = all_results.get(model_name, {}).get("k10_r1", {})
        k10_r5 = all_results.get(model_name, {}).get("k10_r5", {})

        acc_k0_r1 = k0_r1.get("accuracy", 0) * 100
        acc_k0_r5 = k0_r5.get("accuracy", 0) * 100
        acc_k10_r1 = k10_r1.get("accuracy", 0) * 100
        acc_k10_r5 = k10_r5.get("accuracy", 0) * 100

        print(f"| {model_name} | {acc_k0_r1:.1f}% | {acc_k0_r5:.1f}% | {acc_k10_r1:.1f}% | {acc_k10_r5:.1f}% |")

    # Save summary
    summary_file = "eval_results/element_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

    # Generate bar charts
    print("\nGenerating bar charts...")
    plot_bar_chart(all_results)
    plot_zero_shot_comparison(all_results)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
