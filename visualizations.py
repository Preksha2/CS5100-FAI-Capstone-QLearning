"""
CS5100 FAI Capstone – Visualization & Plotting
===============================================
Loads experiment results from results/ and generates all figures for the
capstone report and Milestone 3 check-in.

Figures generated:
    1. Convergence curves      — Q-error decreasing over episodes (verifies theorem)
    2. Training reward curves  — learning progress across all environments
    3. Performance comparison  — bar chart of final policy quality
    4. Overestimation bias     — Q-learning vs Double Q-learning bias
    5. CliffWalking safety     — on-policy vs off-policy cliff falls
    6. Sensitivity analysis    — effect of alpha, gamma, epsilon
    7. Q-error comparison      — final Q-error across all algo x env combinations

Usage:
    python3 visualizations.py
    (Requires results/ directory populated by run_experiments.py)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# Plot style configuration
matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# Consistent colors for each algorithm across all plots
ALGO_COLORS = {
    "Q-learning": "#2196F3",        # Blue
    "SARSA": "#4CAF50",              # Green
    "Double Q-learning": "#FF9800",  # Orange
}
ALGO_ORDER = ["Q-learning", "SARSA", "Double Q-learning"]

RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def load_results():
    """Load the merged experiment results."""
    with open(os.path.join(RESULTS_DIR, "experiment_summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(RESULTS_DIR, "training_curves.json")) as f:
        curves = json.load(f)
    return summary, curves


# =============================================================================
# FIGURE 1: Convergence of Q-values to Q* (FrozenLake-4x4)
# =============================================================================
def plot_convergence(curves):
    """
    This is the most important figure — it directly tests the central theorem
    from Watkins & Dayan (1992): Q_n -> Q* with probability 1 as n -> infinity.

    We plot ||Q - Q*||_inf (max absolute error) over training episodes.
    A decreasing curve confirms empirical convergence.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    conv_data = curves.get("convergence_tracking", {})

    # Left plot: Q-error (log scale)
    for algo_name in ["Q-learning", "SARSA"]:
        if algo_name not in conv_data:
            continue
        checkpoints = conv_data[algo_name]
        episodes = [c["episode"] for c in checkpoints]
        errors = [c["q_error"] for c in checkpoints]
        ax1.plot(episodes, errors, marker="o", markersize=5,
                 color=ALGO_COLORS[algo_name], label=algo_name, linewidth=2)

    ax1.set_xlabel("Training Episodes")
    ax1.set_ylabel("||Q - Q*||_inf (Max Absolute Error)")
    ax1.set_title("Convergence to Optimal Q-Values")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Right plot: Mean reward over training
    for algo_name in ["Q-learning", "SARSA"]:
        if algo_name not in conv_data:
            continue
        checkpoints = conv_data[algo_name]
        episodes = [c["episode"] for c in checkpoints]
        rewards = [c["mean_reward_100"] for c in checkpoints]
        ax2.plot(episodes, rewards, marker="s", markersize=5,
                 color=ALGO_COLORS[algo_name], label=algo_name, linewidth=2)

    ax2.set_xlabel("Training Episodes")
    ax2.set_ylabel("Mean Reward (last 100 episodes)")
    ax2.set_title("Learning Progress")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("FrozenLake-4x4: Empirical Verification of Q-learning Convergence",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_convergence.png"))
    plt.close(fig)
    print("  Saved fig1_convergence.png")


# =============================================================================
# FIGURE 2: Training Reward Curves (All Environments)
# =============================================================================
def plot_training_curves(curves):
    """
    Smoothed average reward over training for each algorithm in each environment.
    Shows: how fast each algorithm learns, and what final performance it reaches.
    """
    env_names = [k for k in curves if k != "convergence_tracking"]
    n_envs = len(env_names)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, env_name in enumerate(env_names):
        ax = axes[idx]
        env_data = curves[env_name]

        for algo_name in ALGO_ORDER:
            if algo_name in env_data and len(env_data[algo_name]) > 0:
                ax.plot(env_data[algo_name],
                        color=ALGO_COLORS[algo_name],
                        label=algo_name, linewidth=1.5, alpha=0.85)

        ax.set_xlabel("Episode (smoothed, window=100)")
        ax.set_ylabel("Average Reward")
        ax.set_title(env_name)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots if fewer than 4 envs
    for idx in range(n_envs, 4):
        axes[idx].set_visible(False)

    fig.suptitle("Training Reward Curves (Mean over 10 seeds, smoothed)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_training_curves.png"))
    plt.close(fig)
    print("  Saved fig2_training_curves.png")


# =============================================================================
# FIGURE 3: Final Performance Comparison (Bar Chart)
# =============================================================================
def plot_performance_comparison(summary):
    """
    Bar chart of mean evaluation reward with 95% confidence intervals.
    This answers: which algorithm produces the best FINAL POLICY?
    """
    env_names = [k for k in summary if k != "sensitivity"]

    fig, axes = plt.subplots(1, len(env_names), figsize=(4 * len(env_names), 5))
    if len(env_names) == 1:
        axes = [axes]

    for idx, env_name in enumerate(env_names):
        ax = axes[idx]
        means, cis, colors, labels = [], [], [], []

        for algo_name in ALGO_ORDER:
            if algo_name in summary.get(env_name, {}):
                r = summary[env_name][algo_name]
                means.append(r["eval_mean_reward"])
                cis.append(r.get("eval_reward_ci95", 0))
                colors.append(ALGO_COLORS[algo_name])
                labels.append(algo_name)

        bars = ax.bar(labels, means, yerr=cis, color=colors, capsize=5,
                      alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.set_title(env_name, fontsize=11)
        ax.set_ylabel("Mean Evaluation Reward")
        ax.grid(True, axis="y", alpha=0.3)

        # Value labels on bars
        for bar, mean_val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f"{mean_val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Final Policy Performance (50 eval episodes x 10 seeds, 95% CI)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_performance_comparison.png"))
    plt.close(fig)
    print("  Saved fig3_performance_comparison.png")


# =============================================================================
# FIGURE 4: Overestimation Bias (Q-learning vs Double Q-learning)
# =============================================================================
def plot_overestimation_bias(summary):
    """
    The max operator in Q-learning systematically overestimates Q-values
    when they are noisy. Double Q-learning (van Hasselt, 2010) fixes this
    by decoupling action selection from value estimation.

    This plot shows mean(Q_learned - Q*) for each algorithm:
        - Positive = overestimation
        - Negative = underestimation
        - Near zero = unbiased
    """
    env_names = [k for k in summary if k != "sensitivity"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(env_names))
    width = 0.25

    for i, algo_name in enumerate(ALGO_ORDER):
        biases = []
        stds = []
        for env_name in env_names:
            r = summary.get(env_name, {}).get(algo_name, {})
            biases.append(r.get("mean_overestimation_bias", 0))
            stds.append(r.get("std_overestimation_bias", 0))
        ax.bar(x + i * width, biases, width, yerr=stds,
               label=algo_name, color=ALGO_COLORS[algo_name],
               alpha=0.8, capsize=4)

    ax.set_xlabel("Environment")
    ax.set_ylabel("Mean Bias: (Q_learned - Q*)")
    ax.set_title("Overestimation Bias Comparison Across Algorithms")
    ax.set_xticks(x + width)
    ax.set_xticklabels(env_names, rotation=15)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_overestimation_bias.png"))
    plt.close(fig)
    print("  Saved fig4_overestimation_bias.png")


# =============================================================================
# FIGURE 5: CliffWalking Safety Comparison
# =============================================================================
def plot_cliff_safety(summary):
    """
    The classic demonstration of on-policy vs off-policy behavior:

    Q-learning (off-policy) learns the OPTIMAL path along the cliff edge
    because it bootstraps from max Q. But during training/evaluation with
    any exploration noise, the agent frequently falls off the cliff.

    SARSA (on-policy) learns the SAFE path along the top row because its
    updates account for the exploration policy. It gets slightly lower
    reward but avoids catastrophic cliff falls.

    This is from Sutton & Barto (2018), Example 6.6.
    """
    cliff_data = summary.get("CliffWalking", {})
    if not cliff_data:
        print("  Skipping fig5 (no CliffWalking data)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    algo_names = [a for a in ALGO_ORDER if a in cliff_data]

    # Left: Cliff falls
    falls = [cliff_data[a].get("mean_cliff_falls", 0) for a in algo_names]
    fall_stds = [cliff_data[a].get("std_cliff_falls", 0) for a in algo_names]
    colors = [ALGO_COLORS[a] for a in algo_names]

    ax1.bar(algo_names, falls, yerr=fall_stds, color=colors, alpha=0.8, capsize=5)
    ax1.set_ylabel("Cliff Falls (per 50 eval episodes)")
    ax1.set_title("Policy Safety: Cliff-Fall Frequency")
    ax1.grid(True, axis="y", alpha=0.3)

    # Right: Reward
    rewards = [cliff_data[a]["eval_mean_reward"] for a in algo_names]
    reward_cis = [cliff_data[a].get("eval_reward_ci95", 0) for a in algo_names]

    ax2.bar(algo_names, rewards, yerr=reward_cis, color=colors, alpha=0.8, capsize=5)
    ax2.set_ylabel("Mean Evaluation Reward")
    ax2.set_title("Reward vs Safety Tradeoff")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("CliffWalking: On-Policy (SARSA) vs Off-Policy (Q-learning) Safety",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig5_cliff_safety.png"))
    plt.close(fig)
    print("  Saved fig5_cliff_safety.png")


# =============================================================================
# FIGURE 6: Hyperparameter Sensitivity
# =============================================================================
def plot_sensitivity(summary):
    """
    The paper proves convergence under specific conditions but gives no
    practical guidance on hyperparameter choice. These plots fill that gap:

    Left:   alpha (learning rate) — too low = slow, too high = unstable
    Center: gamma (discount factor) — higher gamma = harder to converge
    Right:  epsilon (exploration) — too little = miss states, too much = noisy
    """
    sensitivity = summary.get("sensitivity", {})
    if not sensitivity:
        print("  Skipping fig6 (no sensitivity data)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Alpha ---
    ax = axes[0]
    alpha_data = sensitivity.get("alpha_sensitivity", {})
    if alpha_data:
        alphas = sorted(alpha_data.keys(), key=float)
        vals = [float(a) for a in alphas]
        means = [alpha_data[a]["mean_q_error"] for a in alphas]
        stds = [alpha_data[a]["std_q_error"] for a in alphas]
        ax.errorbar(vals, means, yerr=stds, marker="o", capsize=4,
                    color="#2196F3", linewidth=2, markersize=7)
    ax.set_xlabel("Learning Rate (alpha)")
    ax.set_ylabel("Mean Q-Error ||Q - Q*||_inf")
    ax.set_title("Effect of Learning Rate")
    ax.grid(True, alpha=0.3)

    # --- Gamma ---
    ax = axes[1]
    gamma_data = sensitivity.get("gamma_sensitivity", {})
    if gamma_data:
        gammas = sorted(gamma_data.keys(), key=float)
        vals = [float(g) for g in gammas]
        means = [gamma_data[g]["mean_q_error"] for g in gammas]
        stds = [gamma_data[g]["std_q_error"] for g in gammas]
        ax.errorbar(vals, means, yerr=stds, marker="s", capsize=4,
                    color="#4CAF50", linewidth=2, markersize=7)
    ax.set_xlabel("Discount Factor (gamma)")
    ax.set_ylabel("Mean Q-Error ||Q - Q*||_inf")
    ax.set_title("Effect of Discount Factor")
    ax.grid(True, alpha=0.3)

    # --- Epsilon ---
    ax = axes[2]
    eps_data = sensitivity.get("epsilon_sensitivity", {})
    if eps_data:
        epsilons = sorted(eps_data.keys(), key=float)
        vals = [float(e) for e in epsilons]
        means = [eps_data[e]["mean_q_error"] for e in epsilons]
        stds = [eps_data[e]["std_q_error"] for e in epsilons]
        ax.errorbar(vals, means, yerr=stds, marker="^", capsize=4,
                    color="#FF9800", linewidth=2, markersize=7)
    ax.set_xlabel("Exploration Rate (epsilon)")
    ax.set_ylabel("Mean Q-Error ||Q - Q*||_inf")
    ax.set_title("Effect of Exploration Rate")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Hyperparameter Sensitivity — Q-learning on FrozenLake-4x4 (5 seeds)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig6_sensitivity.png"))
    plt.close(fig)
    print("  Saved fig6_sensitivity.png")


# =============================================================================
# FIGURE 7: Q-Error Comparison (All Algo x All Env)
# =============================================================================
def plot_q_error_comparison(summary):
    """
    Final Q-error across all algorithm-environment combinations.
    Lower = algorithm converged closer to Q*.
    """
    env_names = [k for k in summary if k != "sensitivity"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(env_names))
    width = 0.25

    for i, algo_name in enumerate(ALGO_ORDER):
        errors = []
        stds = []
        for env_name in env_names:
            r = summary.get(env_name, {}).get(algo_name, {})
            errors.append(r.get("mean_q_error", 0))
            stds.append(r.get("std_q_error", 0))
        ax.bar(x + i * width, errors, width, yerr=stds,
               label=algo_name, color=ALGO_COLORS[algo_name],
               alpha=0.8, capsize=4)

    ax.set_xlabel("Environment")
    ax.set_ylabel("||Q - Q*||_inf (lower = better)")
    ax.set_title("Final Q-Value Error After Training (10 seeds)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(env_names, rotation=15)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig7_q_error_comparison.png"))
    plt.close(fig)
    print("  Saved fig7_q_error_comparison.png")


# =============================================================================
# RESULTS TABLE (text file for the report)
# =============================================================================
def save_results_table(summary):
    """Print and save a formatted results table."""
    env_names = [k for k in summary if k != "sensitivity"]

    lines = []
    lines.append("=" * 95)
    lines.append("COMPLETE RESULTS TABLE — CS5100 FAI Capstone")
    lines.append("Watkins & Dayan (1992) Q-learning Reproduction + Comparative Study")
    lines.append("=" * 95)
    lines.append("")
    lines.append(f"{'Environment':<18} {'Algorithm':<20} {'Eval Reward':>12} "
                 f"{'95% CI':>8} {'Q-Error':>10} {'Bias':>10} {'Cliff':>8}")
    lines.append("-" * 95)

    for env_name in env_names:
        for algo_name in ALGO_ORDER:
            r = summary.get(env_name, {}).get(algo_name, {})
            if r:
                reward = f"{r['eval_mean_reward']:.3f}"
                ci = f"{r.get('eval_reward_ci95', 0):.3f}"
                q_err = f"{r.get('mean_q_error', 0):.4f}"
                bias = f"{r.get('mean_overestimation_bias', 0):.4f}"
                cliff = f"{r.get('mean_cliff_falls', '--'):>5}" if "mean_cliff_falls" in r else "   --"
                lines.append(f"{env_name:<18} {algo_name:<20} {reward:>12} "
                             f"{ci:>8} {q_err:>10} {bias:>10} {cliff:>8}")
        lines.append("-" * 95)

    # Add sensitivity summary
    sens = summary.get("sensitivity", {})
    if sens:
        lines.append("")
        lines.append("HYPERPARAMETER SENSITIVITY (Q-learning on FrozenLake-4x4)")
        lines.append("-" * 60)

        alpha_data = sens.get("alpha_sensitivity", {})
        if alpha_data:
            lines.append("  Alpha (learning rate):")
            for a in sorted(alpha_data.keys(), key=float):
                d = alpha_data[a]
                lines.append(f"    alpha={float(a):<5} -> Q-error: {d['mean_q_error']:.4f} "
                             f"+/- {d['std_q_error']:.4f}")

        gamma_data = sens.get("gamma_sensitivity", {})
        if gamma_data:
            lines.append("  Gamma (discount factor):")
            for g in sorted(gamma_data.keys(), key=float):
                d = gamma_data[g]
                lines.append(f"    gamma={float(g):<5} -> Q-error: {d['mean_q_error']:.4f} "
                             f"+/- {d['std_q_error']:.4f}")

        eps_data = sens.get("epsilon_sensitivity", {})
        if eps_data:
            lines.append("  Epsilon (exploration rate):")
            for e in sorted(eps_data.keys(), key=float):
                d = eps_data[e]
                lines.append(f"    eps={float(e):<5}   -> Q-error: {d['mean_q_error']:.4f} "
                             f"+/- {d['std_q_error']:.4f}")

    table_text = "\n".join(lines)
    print("\n" + table_text)

    table_path = os.path.join(FIGURES_DIR, "results_table.txt")
    with open(table_path, "w") as f:
        f.write(table_text)
    print(f"\n  Saved {table_path}")


# =============================================================================
# MAIN
# =============================================================================
def generate_all_figures():
    """Load results and generate all figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading experiment results...")
    summary, curves = load_results()

    print("\nGenerating figures...")
    plot_convergence(curves)
    plot_training_curves(curves)
    plot_performance_comparison(summary)
    plot_overestimation_bias(summary)
    plot_cliff_safety(summary)
    plot_sensitivity(summary)
    plot_q_error_comparison(summary)
    save_results_table(summary)

    print(f"\nAll 7 figures + results table saved to {FIGURES_DIR}/")
    print("These can be included directly in your capstone report.")


if __name__ == "__main__":
    generate_all_figures()
