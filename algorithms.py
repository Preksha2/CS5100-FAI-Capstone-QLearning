"""
CS5100 FAI Capstone – TD Control Algorithms
============================================
Selected Paper: Watkins & Dayan (1992). Q-learning. Machine Learning, 8(3–4), 279–292.
Extended with:  Rummery & Niranjan (1994) – SARSA
                van Hasselt (2010) – Double Q-learning

This module implements:
    1. Value Iteration    — computes ground-truth Q* using known dynamics
    2. Q-learning         — off-policy TD control (the paper's algorithm)
    3. SARSA              — on-policy TD control
    4. Double Q-learning  — bias-corrected off-policy TD control
    5. Helper functions   — policy evaluation, Q-error, overestimation bias

All algorithms are TABULAR — they store a Q-value for every (state, action) pair
in a NumPy array. This matches the scope of Watkins & Dayan's convergence proof,
which only applies to the tabular setting.

Key theoretical background:
    - An MDP is defined by (S, A, P, R, γ) — states, actions, transitions, rewards, discount
    - The BELLMAN OPTIMALITY EQUATION defines Q*:
        Q*(s, a) = Σ_s' P(s'|s,a) [R(s,a,s') + γ max_a' Q*(s', a')]
    - Q-learning approximates this without knowing P, using samples instead
    - Convergence requires: all (s,a) visited infinitely often, and learning rate
      satisfies Σα = ∞, Σα² < ∞ (Robbins-Monro conditions)
"""

import numpy as np
from typing import Tuple


# =============================================================================
# 1. VALUE ITERATION — Ground Truth Q* Computation
# =============================================================================
def value_iteration(env, gamma: float = 0.99, theta: float = 1e-8,
                    max_iterations: int = 10000) -> np.ndarray:
    """
    Compute optimal Q-values Q*(s, a) using the Bellman optimality equation.

    This is our GROUND TRUTH. We use it to measure how close the TD algorithms
    get to the true optimal values. Value iteration requires the full transition
    model P(s'|s,a), which is available in our environments as env.P.

    The update at each iteration:
        Q(s, a) = Σ_s' P(s'|s,a) * [R(s,a,s') + γ * max_a' Q(s', a')]

    We iterate until the maximum change across all Q-values falls below theta.

    Parameters
    ----------
    env : environment object
        Must have env.P[s][a] = [(prob, next_state, reward, done), ...]
    gamma : float
        Discount factor ∈ (0, 1). Higher γ = agent cares more about future.
    theta : float
        Convergence threshold. Stop when max |Q_new - Q_old| < theta.
    max_iterations : int
        Safety cap to prevent infinite loops.

    Returns
    -------
    Q_star : np.ndarray, shape (n_states, n_actions)
        The optimal state-action value function.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))

    for iteration in range(max_iterations):
        Q_old = Q.copy()

        for s in range(n_states):
            for a in range(n_actions):
                # Apply Bellman equation using known transition probabilities
                q_sa = 0.0
                for prob, next_state, reward, done in env.P[s][a]:
                    if done:
                        # Terminal state: no future rewards
                        q_sa += prob * reward
                    else:
                        # Immediate reward + discounted best future value
                        q_sa += prob * (reward + gamma * np.max(Q_old[next_state]))
                Q[s, a] = q_sa

        # Check convergence
        delta = np.max(np.abs(Q - Q_old))
        if delta < theta:
            break

    return Q


# =============================================================================
# 2. Q-LEARNING — Off-Policy TD Control (Watkins & Dayan, 1992)
# =============================================================================
def q_learning(env, num_episodes: int = 5000, alpha: float = 0.1,
               gamma: float = 0.99, epsilon: float = 0.1,
               epsilon_decay: float = 0.0, epsilon_min: float = 0.01,
               seed: int = 42) -> Tuple[np.ndarray, list]:
    """
    Tabular Q-learning with ε-greedy exploration.

    THE CORE UPDATE RULE (Equation from Watkins & Dayan, 1992):

        Q(s, a) ← Q(s, a) + α * [ r + γ * max_a' Q(s', a') - Q(s, a) ]
                                    ↑                           ↑
                                TD target                  current estimate

    The key insight: the MAX operator makes this OFF-POLICY. The agent
    bootstraps from the BEST possible next action, regardless of what
    action it actually takes next. This means:
        - Q-learning learns Q* (the optimal Q-values) even while following
          a suboptimal exploration policy
        - But the max also introduces MAXIMIZATION BIAS — it tends to
          overestimate Q-values when they are noisy

    The TD ERROR = [r + γ max Q(s',a')] - Q(s,a) measures how surprised
    the agent is. If positive, the outcome was better than expected.
    If negative, it was worse. Learning gradually reduces this error.

    ε-GREEDY EXPLORATION:
        With probability ε: take random action (explore)
        With probability 1-ε: take greedy action argmax Q(s,a) (exploit)
        ε decays over time: we explore less as we learn more.

    Parameters
    ----------
    env : environment object
        Must have reset(seed) and step(action) methods.
    num_episodes : int
        Total training episodes.
    alpha : float
        Learning rate. Controls how much we update toward the new estimate.
        Too high → unstable. Too low → slow learning.
    gamma : float
        Discount factor ∈ (0, 1). From the paper's convergence conditions.
    epsilon : float
        Initial exploration rate.
    epsilon_decay : float
        Amount to subtract from epsilon each episode (linear decay).
    epsilon_min : float
        Floor for epsilon — always maintain some exploration.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Q : np.ndarray, shape (n_states, n_actions)
        Learned Q-table after training.
    episode_rewards : list of float
        Total reward obtained in each episode (for plotting learning curves).
    """
    rng = np.random.default_rng(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize Q-table to zeros
    # The convergence theorem guarantees Q → Q* regardless of initialization
    Q = np.zeros((n_states, n_actions))

    episode_rewards = []
    current_epsilon = epsilon

    for episode in range(num_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        total_reward = 0.0
        done = False

        while not done:
            # ── ε-Greedy Action Selection ──
            if rng.random() < current_epsilon:
                # EXPLORE: random action
                action = int(rng.integers(0, n_actions))
            else:
                # EXPLOIT: greedy action (break ties randomly)
                q_values = Q[state]
                max_q = np.max(q_values)
                best_actions = np.where(q_values == max_q)[0]
                action = int(rng.choice(best_actions))

            # Take action, observe outcome
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # ── Q-Learning Update ──
            if terminated:
                # Terminal state: no future value
                td_target = reward
            else:
                # Bootstrap from the BEST next action (the MAX)
                td_target = reward + gamma * np.max(Q[next_state])

            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        # Decay exploration rate
        if epsilon_decay > 0:
            current_epsilon = max(epsilon_min, current_epsilon - epsilon_decay)

    return Q, episode_rewards


# =============================================================================
# 3. SARSA — On-Policy TD Control (Rummery & Niranjan, 1994)
# =============================================================================
def sarsa(env, num_episodes: int = 5000, alpha: float = 0.1,
          gamma: float = 0.99, epsilon: float = 0.1,
          epsilon_decay: float = 0.0, epsilon_min: float = 0.01,
          seed: int = 42) -> Tuple[np.ndarray, list]:
    """
    Tabular SARSA (State-Action-Reward-State-Action).

    THE KEY DIFFERENCE FROM Q-LEARNING:

        Q-learning:  Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        SARSA:       Q(s,a) ← Q(s,a) + α * [r + γ * Q(s', a')       - Q(s,a)]
                                                      ^^^^^^^^
                                                      Uses the ACTUAL next action a'
                                                      chosen by the ε-greedy policy,
                                                      NOT the greedy maximum.

    This makes SARSA ON-POLICY: it evaluates and improves the policy it's
    actually following, including the exploration. Consequences:

        1. SARSA is more CONSERVATIVE — on CliffWalking, it learns the safe
           path along the top row because it factors in the risk that
           ε-greedy exploration will occasionally push it off the cliff.

        2. SARSA converges to Q* only if ε → 0 (GLIE condition: Greedy in
           the Limit with Infinite Exploration). With fixed ε, it converges
           to Q_π where π is the ε-greedy policy.

    The name comes from the quintuple used in each update: (S, A, R, S', A').

    Parameters: Same as q_learning.
    Returns: Same as q_learning.
    """
    rng = np.random.default_rng(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    episode_rewards = []
    current_epsilon = epsilon

    def choose_action(state, eps):
        """ε-greedy action selection."""
        if rng.random() < eps:
            return int(rng.integers(0, n_actions))
        else:
            q_values = Q[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return int(rng.choice(best_actions))

    for episode in range(num_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 2**31)))

        # SARSA difference #1: choose the first action BEFORE the loop
        action = choose_action(state, current_epsilon)

        total_reward = 0.0
        done = False

        while not done:
            # Take action A, observe R, S'
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # SARSA difference #2: choose next action A' using SAME policy
            if not done:
                next_action = choose_action(next_state, current_epsilon)
            else:
                next_action = 0  # Won't be used

            # ── SARSA Update ──
            if terminated:
                td_target = reward
            else:
                # Use Q(S', A') — the ACTUAL next action, not max
                td_target = reward + gamma * Q[next_state, next_action]

            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            # SARSA difference #3: carry forward both S' AND A'
            state = next_state
            action = next_action
            total_reward += reward

        episode_rewards.append(total_reward)

        if epsilon_decay > 0:
            current_epsilon = max(epsilon_min, current_epsilon - epsilon_decay)

    return Q, episode_rewards


# =============================================================================
# 4. DOUBLE Q-LEARNING — Bias-Corrected TD Control (van Hasselt, 2010)
# =============================================================================
def double_q_learning(env, num_episodes: int = 5000, alpha: float = 0.1,
                      gamma: float = 0.99, epsilon: float = 0.1,
                      epsilon_decay: float = 0.0, epsilon_min: float = 0.01,
                      seed: int = 42) -> Tuple[np.ndarray, list]:
    """
    Double Q-learning with ε-greedy exploration.

    THE MAXIMIZATION BIAS PROBLEM:
        In standard Q-learning, the target uses max_a' Q(s', a').
        When Q-values are noisy estimates (which they always are during learning),
        the max of noisy values is biased UPWARD. Intuitively: if you have 10
        noisy estimates of the same true value, the maximum will on average be
        higher than the true value.

        This causes Q-learning to systematically OVERESTIMATE Q-values,
        which can lead to suboptimal policies in stochastic environments.

    THE SOLUTION — TWO INDEPENDENT Q-TABLES:
        Maintain Q_A and Q_B. On each step, flip a coin:
            If heads (update Q_A):
                a* = argmax_a Q_A(s', a)     ← use Q_A to SELECT the best action
                target = r + γ * Q_B(s', a*) ← use Q_B to EVALUATE that action
                Update Q_A(s, a) toward the target

            If tails (update Q_B):
                a* = argmax_a Q_B(s', a)     ← use Q_B to SELECT
                target = r + γ * Q_A(s', a*) ← use Q_A to EVALUATE
                Update Q_B(s, a)

        By DECOUPLING action selection from action evaluation, the positive
        bias is eliminated. The table that selects the action is independent
        of the table that estimates its value.

    For ε-greedy during training, we use Q_A + Q_B (sum ≡ ranking by average).
    The final Q-table is the average: (Q_A + Q_B) / 2.

    Parameters: Same as q_learning.
    Returns: Same as q_learning (returns averaged Q-table).
    """
    rng = np.random.default_rng(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Two independent Q-tables — this is the key innovation
    Q_A = np.zeros((n_states, n_actions))
    Q_B = np.zeros((n_states, n_actions))

    episode_rewards = []
    current_epsilon = epsilon

    for episode in range(num_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        total_reward = 0.0
        done = False

        while not done:
            # ε-greedy using SUM of both tables (equivalent to average ranking)
            if rng.random() < current_epsilon:
                action = int(rng.integers(0, n_actions))
            else:
                combined_q = Q_A[state] + Q_B[state]
                max_q = np.max(combined_q)
                best_actions = np.where(combined_q == max_q)[0]
                action = int(rng.choice(best_actions))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # ── Double Q-Learning Update ──
            # Randomly choose which table to update (coin flip)
            if rng.random() < 0.5:
                # Update Q_A: SELECT with Q_A, EVALUATE with Q_B
                if terminated:
                    td_target = reward
                else:
                    best_next = np.argmax(Q_A[next_state])         # SELECT
                    td_target = reward + gamma * Q_B[next_state, best_next]  # EVALUATE

                Q_A[state, action] += alpha * (td_target - Q_A[state, action])
            else:
                # Update Q_B: SELECT with Q_B, EVALUATE with Q_A
                if terminated:
                    td_target = reward
                else:
                    best_next = np.argmax(Q_B[next_state])         # SELECT
                    td_target = reward + gamma * Q_A[next_state, best_next]  # EVALUATE

                Q_B[state, action] += alpha * (td_target - Q_B[state, action])

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        if epsilon_decay > 0:
            current_epsilon = max(epsilon_min, current_epsilon - epsilon_decay)

    # Return the AVERAGE of both tables as the final estimate
    Q_combined = (Q_A + Q_B) / 2.0
    return Q_combined, episode_rewards


# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================
def extract_policy(Q: np.ndarray) -> np.ndarray:
    """
    Extract the greedy policy from a Q-table.
    π*(s) = argmax_a Q(s, a)

    Returns an array of length n_states, where each entry is the best action.
    """
    return np.argmax(Q, axis=1)


def evaluate_policy(env, Q: np.ndarray, num_episodes: int = 100,
                    seed: int = 0) -> Tuple[float, float, int]:
    """
    Evaluate the greedy policy derived from a Q-table.

    Runs the greedy policy (no exploration, ε=0) for num_episodes and
    computes statistics. This tells us how good the LEARNED policy actually is.

    Returns
    -------
    mean_reward : float
        Average total reward across evaluation episodes.
    std_reward : float
        Standard deviation of rewards (measures consistency).
    cliff_falls : int
        Number of times the agent received -100 reward (CliffWalking metric).
    """
    rng = np.random.default_rng(seed)
    rewards = []
    cliff_falls = 0

    for ep in range(num_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 500:
            # Greedy action — no exploration
            action = int(np.argmax(Q[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if reward == -100:  # Cliff fall detection
                cliff_falls += 1

            state = next_state
            steps += 1

        rewards.append(total_reward)

    return float(np.mean(rewards)), float(np.std(rewards)), cliff_falls


def compute_q_error(Q_learned: np.ndarray, Q_star: np.ndarray) -> float:
    """
    Compute ||Q_learned - Q*||_∞  (max absolute error across all state-action pairs).

    This is the primary metric for verifying convergence.
    The paper's theorem states Q_n → Q* as n → ∞, so this error should
    decrease over training. We use the infinity norm because it captures
    the WORST-CASE error.
    """
    return float(np.max(np.abs(Q_learned - Q_star)))


def compute_overestimation_bias(Q_learned: np.ndarray, Q_star: np.ndarray) -> float:
    """
    Compute mean overestimation bias: mean(Q_learned - Q*).

    Positive values indicate OVERESTIMATION — the agent thinks states are
    more valuable than they really are. This is the bias that Double
    Q-learning is designed to fix.

    We expect:
        Q-learning        → positive bias (overestimates due to max)
        Double Q-learning → near-zero or smaller bias
        SARSA             → negative bias (underestimates due to exploration)
    """
    return float(np.mean(Q_learned - Q_star))


def smooth_rewards(rewards: list, window: int = 100) -> np.ndarray:
    """
    Smooth a reward curve using a rolling average.
    This makes learning curves easier to read by reducing noise.

    Parameters
    ----------
    rewards : list
        Raw per-episode rewards.
    window : int
        Size of the smoothing window.

    Returns
    -------
    smoothed : np.ndarray
        Smoothed reward curve (shorter by window-1 elements).
    """
    if len(rewards) < window:
        return np.array(rewards)
    return np.convolve(rewards, np.ones(window) / window, mode='valid')


# =============================================================================
# QUICK TEST — run this file directly to verify algorithms work
# =============================================================================
if __name__ == "__main__":
    from environments import make_env

    print("Testing algorithms on FrozenLake-4x4...\n")

    env = make_env("FrozenLake-v1", map_name="4x4", is_slippery=True)

    # 1. Value Iteration
    print("1. Value Iteration (ground truth)...")
    Q_star = value_iteration(env, gamma=0.99)
    print(f"   Q* range: [{Q_star.min():.4f}, {Q_star.max():.4f}]")
    policy = extract_policy(Q_star)
    action_names = ['←', '↓', '→', '↑']
    print(f"   Optimal policy: {[action_names[a] for a in policy]}")

    # 2. Q-learning
    print("\n2. Q-learning (5000 episodes)...")
    env = make_env("FrozenLake-v1", map_name="4x4", is_slippery=True)
    Q_ql, rewards_ql = q_learning(env, num_episodes=5000, seed=42)
    err_ql = compute_q_error(Q_ql, Q_star)
    bias_ql = compute_overestimation_bias(Q_ql, Q_star)
    print(f"   Q-error: {err_ql:.4f}")
    print(f"   Overestimation bias: {bias_ql:.4f}")
    print(f"   Last 100 episodes avg reward: {np.mean(rewards_ql[-100:]):.3f}")

    # 3. SARSA
    print("\n3. SARSA (5000 episodes)...")
    env = make_env("FrozenLake-v1", map_name="4x4", is_slippery=True)
    Q_sa, rewards_sa = sarsa(env, num_episodes=5000, seed=42)
    err_sa = compute_q_error(Q_sa, Q_star)
    bias_sa = compute_overestimation_bias(Q_sa, Q_star)
    print(f"   Q-error: {err_sa:.4f}")
    print(f"   Overestimation bias: {bias_sa:.4f}")
    print(f"   Last 100 episodes avg reward: {np.mean(rewards_sa[-100:]):.3f}")

    # 4. Double Q-learning
    print("\n4. Double Q-learning (5000 episodes)...")
    env = make_env("FrozenLake-v1", map_name="4x4", is_slippery=True)
    Q_dq, rewards_dq = double_q_learning(env, num_episodes=5000, seed=42)
    err_dq = compute_q_error(Q_dq, Q_star)
    bias_dq = compute_overestimation_bias(Q_dq, Q_star)
    print(f"   Q-error: {err_dq:.4f}")
    print(f"   Overestimation bias: {bias_dq:.4f}")
    print(f"   Last 100 episodes avg reward: {np.mean(rewards_dq[-100:]):.3f}")

    # 5. Policy evaluation
    print("\n5. Policy evaluation (Q-learning, 100 greedy episodes)...")
    env = make_env("FrozenLake-v1", map_name="4x4", is_slippery=True)
    mean_r, std_r, _ = evaluate_policy(env, Q_ql, num_episodes=100)
    print(f"   Mean reward: {mean_r:.3f} ± {std_r:.3f}")

    print("\nAll algorithms working!")
