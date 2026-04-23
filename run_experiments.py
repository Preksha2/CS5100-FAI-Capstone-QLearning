"""
CS5100 FAI Capstone – Experiment Runner
=======================================
Runs the complete experimental pipeline from Milestone 2:
    - 3 algorithms x 4 environments x 10 seeds
    - Convergence tracking (Q-error over training)
    - Hyperparameter sensitivity analysis (alpha, gamma, epsilon)
    - All results saved to results/ as JSON for visualization

Usage:
    python3 run_experiments.py          # Run all phases sequentially
    python3 run_experiments.py 1        # Run only phase 1
    python3 run_experiments.py 2        # Run only phase 2, etc.

Phases:
    1 - Compute ground-truth Q* via value iteration (~2 seconds)
    2 - Main experiments: all algorithms x all environments x 10 seeds
    3 - Convergence tracking: Q-error at checkpoints during training
    4 - Hyperparameter sensitivity analysis (alpha, gamma, epsilon)
    5 - Merge all results into final JSON files for visualization
"""

import sys
import numpy as np
import json
import os
import time
from environments import make_env
from algorithms import (
    value_iteration, q_learning, sarsa, double_q_learning,
    evaluate_policy, compute_q_error, compute_overestimation_bias,
    smooth_rewards,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 0.1
EPSILON_MIN = 0.01

NUM_SEEDS = 10
SENSITIVITY_SEEDS = 5
CONVERGENCE_SEEDS = 3
EVAL_EPISODES = 50

ENVIRONMENTS = {
    "FrozenLake-4x4": {
        "id": "FrozenLake-v1",
        "kwargs": {"map_name": "4x4", "is_slippery": True},
        "num_episodes": 5000,
        "description": "4x4 stochastic gridworld (baseline convergence test)",
    },
    "FrozenLake-8x8": {
        "id": "FrozenLake-v1",
        "kwargs": {"map_name": "8x8", "is_slippery": True},
        "num_episodes": 5000,
        "description": "8x8 stochastic gridworld (scalability + exploration difficulty)",
    },
    "CliffWalking": {
        "id": "CliffWalking-v0",
        "kwargs": {},
        "num_episodes": 2000,
        "description": "Deterministic gridworld with cliff (on-policy vs off-policy safety)",
    },
    "Taxi": {
        "id": "Taxi-v3",
        "kwargs": {},
        "num_episodes": 3000,
        "description": "500-state pickup/dropoff task (sample efficiency)",
    },
}

ALGORITHMS = {
    "Q-learning": q_learning,
    "SARSA": sarsa,
    "Double Q-learning": double_q_learning,
}

RESULTS_DIR = "results"

# =============================================================================
# HELPER: epsilon-greedy with random tie-breaking
# =============================================================================
def eps_greedy_action(Q, state, n_actions, epsilon, rng):
    """
    Choose action using epsilon-greedy with random tie-breaking.
    IMPORTANT: When Q-values are all equal (e.g. all zeros at start),
    np.argmax always returns 0. This traps the agent. Random tie-breaking
    ensures uniform selection among tied actions.
    """
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))
    q_vals = Q[state]
    max_q = np.max(q_vals)
    best = np.where(q_vals == max_q)[0]
    return int(rng.choice(best))

# =============================================================================
# HELPER: Run one algo on one env for one seed
# =============================================================================
def run_single(algo_fn, env_id, env_kwargs, num_episodes, Q_star, seed):
    env = make_env(env_id, **env_kwargs)
    eps_decay = (EPSILON - EPSILON_MIN) / num_episodes
    Q, episode_rewards = algo_fn(
        env, num_episodes=num_episodes,
        alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
        epsilon_decay=eps_decay, epsilon_min=EPSILON_MIN, seed=seed,
    )
    env.close()
    result = {"episode_rewards": episode_rewards}
    if Q_star is not None:
        result["q_error"] = float(compute_q_error(Q, Q_star))
        result["overestimation_bias"] = float(compute_overestimation_bias(Q, Q_star))
    env_eval = make_env(env_id, **env_kwargs)
    mean_r, std_r, cliff_falls = evaluate_policy(env_eval, Q, EVAL_EPISODES, seed)
    env_eval.close()
    result["eval_mean_reward"] = float(mean_r)
    result["eval_std_reward"] = float(std_r)
    result["cliff_falls"] = int(cliff_falls)
    return result

# =============================================================================
# HELPER: Aggregate results across seeds
# =============================================================================
def aggregate_seeds(seed_results, env_name):
    n = len(seed_results)
    agg = {
        "eval_mean_reward": float(np.mean([r["eval_mean_reward"] for r in seed_results])),
        "eval_std_reward": float(np.std([r["eval_mean_reward"] for r in seed_results])),
        "eval_reward_ci95": float(
            1.96 * np.std([r["eval_mean_reward"] for r in seed_results]) / np.sqrt(n)
        ),
    }
    if "q_error" in seed_results[0]:
        agg["mean_q_error"] = float(np.mean([r["q_error"] for r in seed_results]))
        agg["std_q_error"] = float(np.std([r["q_error"] for r in seed_results]))
        agg["mean_overestimation_bias"] = float(np.mean([r["overestimation_bias"] for r in seed_results]))
        agg["std_overestimation_bias"] = float(np.std([r["overestimation_bias"] for r in seed_results]))
    if "CliffWalking" in env_name:
        agg["mean_cliff_falls"] = float(np.mean([r["cliff_falls"] for r in seed_results]))
        agg["std_cliff_falls"] = float(np.std([r["cliff_falls"] for r in seed_results]))
    all_curves = np.array([r["episode_rewards"] for r in seed_results])
    mean_curve = np.mean(all_curves, axis=0)
    agg["smoothed_curve"] = smooth_rewards(mean_curve.tolist(), window=100).tolist()
    return agg

# =============================================================================
# PHASE 1
# =============================================================================
def phase1():
    print("=" * 65)
    print("PHASE 1: Computing Q* via Value Iteration")
    print("=" * 65)
    qstars = {}
    for env_name, cfg in ENVIRONMENTS.items():
        env = make_env(cfg["id"], **cfg["kwargs"])
        t0 = time.time()
        Q_star = value_iteration(env, gamma=GAMMA)
        elapsed = time.time() - t0
        env.close()
        qstars[env_name] = Q_star.tolist()
        print(f"  {env_name}: shape={Q_star.shape}, "
              f"range=[{Q_star.min():.4f}, {Q_star.max():.4f}], time={elapsed:.2f}s")
    path = os.path.join(RESULTS_DIR, "qstars.json")
    with open(path, "w") as f:
        json.dump(qstars, f)
    print(f"\nSaved -> {path}")

# =============================================================================
# PHASE 2
# =============================================================================
def phase2():
    print("=" * 65)
    print(f"PHASE 2: Main Experiments ({len(ALGORITHMS)} algos x "
          f"{len(ENVIRONMENTS)} envs x {NUM_SEEDS} seeds)")
    print("=" * 65)
    with open(os.path.join(RESULTS_DIR, "qstars.json")) as f:
        qstars = {k: np.array(v) for k, v in json.load(f).items()}
    results = {}
    for env_name, cfg in ENVIRONMENTS.items():
        print(f"\n-- {env_name}: {cfg['description']} --")
        results[env_name] = {}
        Q_star = qstars[env_name]
        for algo_name, algo_fn in ALGORITHMS.items():
            print(f"  {algo_name} ({NUM_SEEDS} seeds)...", end=" ", flush=True)
            t0 = time.time()
            seed_results = [
                run_single(algo_fn, cfg["id"], cfg["kwargs"],
                           cfg["num_episodes"], Q_star, seed)
                for seed in range(NUM_SEEDS)
            ]
            agg = aggregate_seeds(seed_results, env_name)
            results[env_name][algo_name] = agg
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)")
            print(f"    Eval reward: {agg['eval_mean_reward']:.3f} "
                  f"+/- {agg['eval_reward_ci95']:.3f} (95% CI)")
            print(f"    Q-error:     {agg.get('mean_q_error', 'N/A'):.4f} "
                  f"+/- {agg.get('std_q_error', 0):.4f}")
            print(f"    Bias:        {agg.get('mean_overestimation_bias', 'N/A'):.4f}")
            if "mean_cliff_falls" in agg:
                print(f"    Cliff falls: {agg['mean_cliff_falls']:.1f}")
    path = os.path.join(RESULTS_DIR, "phase2_main.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {path}")

# =============================================================================
# PHASE 3
# =============================================================================
def phase3():
    print("=" * 65)
    print("PHASE 3: Convergence Tracking (FrozenLake-4x4)")
    print("=" * 65)
    with open(os.path.join(RESULTS_DIR, "qstars.json")) as f:
        qstars = {k: np.array(v) for k, v in json.load(f).items()}
    Q_star = qstars["FrozenLake-4x4"]
    cfg = ENVIRONMENTS["FrozenLake-4x4"]
    num_episodes = cfg["num_episodes"]
    checkpoint_episodes = sorted(set(
        [max(1, int(f * num_episodes)) for f in
         [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]]
    ))
    results = {}
    for algo_name in ["Q-learning", "SARSA"]:
        print(f"\n  {algo_name} ({CONVERGENCE_SEEDS} seeds)...", end=" ", flush=True)
        t0 = time.time()
        all_seed_checkpoints = []
        for seed in range(CONVERGENCE_SEEDS):
            rng = np.random.default_rng(seed)
            env = make_env(cfg["id"], **cfg["kwargs"])
            n_states = env.observation_space.n
            n_actions = env.action_space.n
            Q = np.zeros((n_states, n_actions))
            current_epsilon = EPSILON
            eps_decay = (EPSILON - EPSILON_MIN) / num_episodes
            rewards_so_far = []
            checkpoints = []
            for ep in range(1, num_episodes + 1):
                state, _ = env.reset(seed=int(rng.integers(0, 2**31)))
                done = False
                total_reward = 0.0
                if algo_name == "Q-learning":
                    while not done:
                        action = eps_greedy_action(Q, state, n_actions, current_epsilon, rng)
                        ns, r, term, trunc, _ = env.step(action)
                        done = term or trunc
                        td_target = r if term else r + GAMMA * np.max(Q[ns])
                        Q[state, action] += ALPHA * (td_target - Q[state, action])
                        state = ns
                        total_reward += r
                else:
                    action = eps_greedy_action(Q, state, n_actions, current_epsilon, rng)
                    while not done:
                        ns, r, term, trunc, _ = env.step(action)
                        done = term or trunc
                        na = eps_greedy_action(Q, ns, n_actions, current_epsilon, rng) if not done else 0
                        td_target = r if term else r + GAMMA * Q[ns, na]
                        Q[state, action] += ALPHA * (td_target - Q[state, action])
                        state = ns
                        action = na
                        total_reward += r
                rewards_so_far.append(total_reward)
                current_epsilon = max(EPSILON_MIN, current_epsilon - eps_decay)
                if ep in checkpoint_episodes:
                    recent = (float(np.mean(rewards_so_far[-100:]))
                              if len(rewards_so_far) >= 100
                              else float(np.mean(rewards_so_far)))
                    checkpoints.append({
                        "episode": ep,
                        "q_error": float(np.max(np.abs(Q - Q_star))),
                        "mean_reward_100": recent,
                    })
            env.close()
            all_seed_checkpoints.append(checkpoints)
        avg_checkpoints = []
        for i in range(len(all_seed_checkpoints[0])):
            ep = all_seed_checkpoints[0][i]["episode"]
            avg_err = float(np.mean([s[i]["q_error"] for s in all_seed_checkpoints]))
            avg_rew = float(np.mean([s[i]["mean_reward_100"] for s in all_seed_checkpoints]))
            avg_checkpoints.append({"episode": ep, "q_error": avg_err, "mean_reward_100": avg_rew})
        results[algo_name] = avg_checkpoints
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        print(f"    Initial Q-error: {avg_checkpoints[0]['q_error']:.4f}")
        print(f"    Final Q-error:   {avg_checkpoints[-1]['q_error']:.4f}")
    path = os.path.join(RESULTS_DIR, "phase3_convergence.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {path}")

# =============================================================================
# PHASE 4
# =============================================================================
def phase4():
    print("=" * 65)
    print("PHASE 4: Hyperparameter Sensitivity (FrozenLake-4x4)")
    print("=" * 65)
    with open(os.path.join(RESULTS_DIR, "qstars.json")) as f:
        qstars = {k: np.array(v) for k, v in json.load(f).items()}
    Q_star = qstars["FrozenLake-4x4"]
    cfg = ENVIRONMENTS["FrozenLake-4x4"]
    num_ep = 3000
    results = {}
    print("\n  Varying alpha (learning rate)...")
    results["alpha_sensitivity"] = {}
    for alpha in [0.01, 0.05, 0.1, 0.3, 0.5]:
        errors, rewards = [], []
        for seed in range(SENSITIVITY_SEEDS):
            env = make_env(cfg["id"], **cfg["kwargs"])
            Q, ep_r = q_learning(env, num_ep, alpha=alpha, gamma=0.99, epsilon=0.1, seed=seed)
            env.close()
            errors.append(compute_q_error(Q, Q_star))
            rewards.append(float(np.mean(ep_r[-500:])))
        results["alpha_sensitivity"][str(alpha)] = {
            "mean_q_error": float(np.mean(errors)), "std_q_error": float(np.std(errors)),
            "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
        }
        print(f"    alpha={alpha}: Q-error={np.mean(errors):.4f} +/- {np.std(errors):.4f}")
    print("\n  Varying gamma (discount factor)...")
    results["gamma_sensitivity"] = {}
    for gamma in [0.9, 0.95, 0.99, 0.999]:
        env = make_env(cfg["id"], **cfg["kwargs"])
        Q_star_g = value_iteration(env, gamma=gamma)
        env.close()
        errors, rewards = [], []
        for seed in range(SENSITIVITY_SEEDS):
            env = make_env(cfg["id"], **cfg["kwargs"])
            Q, ep_r = q_learning(env, num_ep, alpha=0.1, gamma=gamma, epsilon=0.1, seed=seed)
            env.close()
            errors.append(compute_q_error(Q, Q_star_g))
            rewards.append(float(np.mean(ep_r[-500:])))
        results["gamma_sensitivity"][str(gamma)] = {
            "mean_q_error": float(np.mean(errors)), "std_q_error": float(np.std(errors)),
            "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
        }
        print(f"    gamma={gamma}: Q-error={np.mean(errors):.4f} +/- {np.std(errors):.4f}")
    print("\n  Varying epsilon (exploration rate)...")
    results["epsilon_sensitivity"] = {}
    for eps in [0.01, 0.05, 0.1, 0.2, 0.3]:
        errors, rewards = [], []
        for seed in range(SENSITIVITY_SEEDS):
            env = make_env(cfg["id"], **cfg["kwargs"])
            Q, ep_r = q_learning(env, num_ep, alpha=0.1, gamma=0.99, epsilon=eps, seed=seed)
            env.close()
            errors.append(compute_q_error(Q, Q_star))
            rewards.append(float(np.mean(ep_r[-500:])))
        results["epsilon_sensitivity"][str(eps)] = {
            "mean_q_error": float(np.mean(errors)), "std_q_error": float(np.std(errors)),
            "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
        }
        print(f"    epsilon={eps}: Q-error={np.mean(errors):.4f} +/- {np.std(errors):.4f}")
    path = os.path.join(RESULTS_DIR, "phase4_sensitivity.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {path}")

# =============================================================================
# PHASE 5
# =============================================================================
def phase5():
    print("=" * 65)
    print("PHASE 5: Merging All Results")
    print("=" * 65)
    with open(os.path.join(RESULTS_DIR, "phase2_main.json")) as f:
        main_results = json.load(f)
    with open(os.path.join(RESULTS_DIR, "phase3_convergence.json")) as f:
        convergence = json.load(f)
    with open(os.path.join(RESULTS_DIR, "phase4_sensitivity.json")) as f:
        sensitivity = json.load(f)
    summary = {}
    curves = {}
    for env_name in ENVIRONMENTS:
        summary[env_name] = {}
        curves[env_name] = {}
        for algo_name in ALGORITHMS:
            data = main_results[env_name][algo_name].copy()
            curves[env_name][algo_name] = data.pop("smoothed_curve", [])
            summary[env_name][algo_name] = data
    summary["sensitivity"] = sensitivity
    curves["convergence_tracking"] = convergence
    summary_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    curves_path = os.path.join(RESULTS_DIR, "training_curves.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(curves_path, "w") as f:
        json.dump(curves, f)
    print(f"  Saved -> {summary_path}")
    print(f"  Saved -> {curves_path}")
    print("\n" + "=" * 90)
    print(f"{'Environment':<18} {'Algorithm':<20} {'Eval Reward':>12} "
          f"{'Q-Error':>10} {'Bias':>10} {'Cliff Falls':>12}")
    print("-" * 90)
    for env_name in ENVIRONMENTS:
        for algo_name in ALGORITHMS:
            r = summary[env_name][algo_name]
            reward = f"{r['eval_mean_reward']:.3f}"
            q_err = f"{r.get('mean_q_error', 0):.4f}"
            bias = f"{r.get('mean_overestimation_bias', 0):.4f}"
            cliff = f"{r.get('mean_cliff_falls', '--'):>6}" if 'mean_cliff_falls' in r else f"{'--':>6}"
            print(f"{env_name:<18} {algo_name:<20} {reward:>12} "
                  f"{q_err:>10} {bias:>10} {cliff:>12}")
        print("-" * 90)

# =============================================================================
# ENTRY POINT
# =============================================================================
def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    phase1(); print()
    phase2(); print()
    phase3(); print()
    phase4(); print()
    phase5()
    print("\n" + "=" * 65)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 65)
    print("Next step: python3 visualizations.py")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if len(sys.argv) > 1:
        phase_num = int(sys.argv[1])
        phases = {1: phase1, 2: phase2, 3: phase3, 4: phase4, 5: phase5}
        if phase_num in phases:
            phases[phase_num]()
        else:
            print(f"Unknown phase {phase_num}. Valid: 1-5")
    else:
        run_all()
