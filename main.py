"""
CS5100 FAI Capstone – Main Entry Point
=======================================
Paper: Watkins & Dayan (1992). Q-learning. Machine Learning, 8(3-4), 279-292.
Extended: SARSA (Rummery & Niranjan, 1994), Double Q-learning (van Hasselt, 2010)

This is the single entry point for the entire capstone project.
Run this file and it will:
    1. Compute ground-truth Q* for all environments
    2. Train Q-learning, SARSA, and Double Q-learning across 4 environments
    3. Track convergence over training (verifies the paper's theorem)
    4. Run hyperparameter sensitivity analysis
    5. Generate all 7 figures + results table

Usage:
    python3 main.py              # Run everything
    python3 main.py --plots-only # Skip experiments, just regenerate figures

Total runtime: ~10-15 minutes on a typical laptop.

Project structure:
    environments.py    — FrozenLake, CliffWalking, Taxi implementations
    algorithms.py      — Value iteration, Q-learning, SARSA, Double Q-learning
    run_experiments.py — Experiment runner (5 phases, saves JSON results)
    visualizations.py  — Generates all figures from saved results
    main.py            — This file (entry point)
    results/           — JSON experiment results (created automatically)
    figures/           — PNG figures + results table (created automatically)
"""

import sys
import os
import time


def main():
    start = time.time()

    plots_only = "--plots-only" in sys.argv

    if not plots_only:
        # ── Run all experiments ──
        print("=" * 65)
        print("  CS5100 FAI CAPSTONE PROJECT")
        print("  Tabular TD Control: Q-learning, SARSA, Double Q-learning")
        print("  Paper: Watkins & Dayan (1992)")
        print("=" * 65)
        print()

        from run_experiments import run_all
        run_all()
    else:
        print("Skipping experiments (--plots-only mode)")
        # Verify results exist
        required = [
            "results/experiment_summary.json",
            "results/training_curves.json",
        ]
        for f in required:
            if not os.path.exists(f):
                print(f"ERROR: {f} not found. Run without --plots-only first.")
                sys.exit(1)

    # ── Generate all figures ──
    print()
    from visualizations import generate_all_figures
    generate_all_figures()

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print()
    print("=" * 65)
    print(f"  DONE! Total time: {minutes}m {seconds}s")
    print("=" * 65)
    print()
    print("Output files:")
    print("  results/experiment_summary.json  — all metrics")
    print("  results/training_curves.json     — reward curves + convergence")
    print("  figures/fig1_convergence.png      — Q-error convergence (KEY FIGURE)")
    print("  figures/fig2_training_curves.png  — learning curves all envs")
    print("  figures/fig3_performance_comparison.png")
    print("  figures/fig4_overestimation_bias.png")
    print("  figures/fig5_cliff_safety.png")
    print("  figures/fig6_sensitivity.png")
    print("  figures/fig7_q_error_comparison.png")
    print("  figures/results_table.txt         — formatted results for report")
    print()
    print("For your Milestone 3 check-in, be ready to discuss:")
    print("  1. The Q-learning update rule and why max makes it off-policy")
    print("  2. How SARSA differs (uses actual next action, not max)")
    print("  3. Why Double Q-learning fixes overestimation bias")
    print("  4. The convergence plot (fig1) — direct verification of the theorem")
    print("  5. CliffWalking safety (fig5) — SARSA safe path vs Q-learning optimal path")
    print("  6. Sensitivity results — why alpha=0.3 works best, why high gamma is harder")


if __name__ == "__main__":
    main()
