"""
CS5100 FAI Capstone – Custom Environment Implementations
=========================================================
Since we can't install Gymnasium in this environment, we implement the four
tabular environments ourselves. Each environment matches the Gymnasium API:
    - reset(seed) → (state, info)
    - step(action) → (next_state, reward, terminated, truncated, info)
    - observation_space.n → number of states
    - action_space.n → number of actions
    - P[s][a] → list of (probability, next_state, reward, done) tuples

Environments:
    1. FrozenLake (4×4 and 8×8) — stochastic gridworld
    2. CliffWalking — deterministic gridworld with cliff penalty
    3. Taxi — pickup/dropoff task with 500 states

These are standard RL benchmarks from Sutton & Barto (2018) and OpenAI Gym.
"""

import numpy as np
from collections import defaultdict


# =============================================================================
# Simple Discrete Space (mimics gymnasium.spaces.Discrete)
# =============================================================================
class DiscreteSpace:
    """Minimal implementation of gymnasium.spaces.Discrete."""
    def __init__(self, n):
        self.n = n


# =============================================================================
# 1. FROZEN LAKE
# =============================================================================
# Standard maps from Gymnasium source code
FROZEN_LAKE_MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}


class FrozenLake:
    """
    FrozenLake-v1 environment.

    The agent navigates a frozen lake grid trying to reach the Goal without
    falling into Holes. The lake is slippery, so the agent doesn't always
    move in the intended direction.

    Tile types:
        S = Start (safe)
        F = Frozen (safe)
        H = Hole (terminal, reward = 0)
        G = Goal (terminal, reward = 1)

    Actions: 0=Left, 1=Down, 2=Right, 3=Up

    Slippery dynamics (is_slippery=True):
        When the agent chooses an action, it moves in the intended direction
        with probability 1/3, and in each perpendicular direction with
        probability 1/3. For example, choosing "Right" results in:
            - 1/3 chance: move Right (intended)
            - 1/3 chance: move Up (perpendicular)
            - 1/3 chance: move Down (perpendicular)

        This stochasticity is what makes FrozenLake challenging — the agent
        must learn a policy that accounts for the randomness in transitions.
        It also means Q-learning needs many episodes to converge because
        the same action from the same state can lead to different outcomes.

    State encoding: state = row * ncol + col (integer from 0 to nrow*ncol - 1)
    """

    def __init__(self, map_name="4x4", is_slippery=True):
        self.desc = np.array([list(row) for row in FROZEN_LAKE_MAPS[map_name]])
        self.nrow, self.ncol = self.desc.shape
        self.is_slippery = is_slippery

        n_states = self.nrow * self.ncol
        n_actions = 4

        self.observation_space = DiscreteSpace(n_states)
        self.action_space = DiscreteSpace(n_actions)

        # Find start state
        self.start_state = None
        for r in range(self.nrow):
            for c in range(self.ncol):
                if self.desc[r, c] == 'S':
                    self.start_state = r * self.ncol + c

        # Build full transition model P[s][a] = [(prob, next_state, reward, done), ...]
        # This is what value iteration uses to compute Q*
        self.P = self._build_transitions()

        self.state = self.start_state
        self._rng = np.random.default_rng()

    def _to_state(self, row, col):
        """Convert (row, col) grid position to flat state index."""
        return row * self.ncol + col

    def _move(self, row, col, action):
        """
        Compute next grid position given current position and action.
        Clips at grid boundaries (hitting a wall keeps you in place).
        """
        if action == 0:    # Left
            col = max(col - 1, 0)
        elif action == 1:  # Down
            row = min(row + 1, self.nrow - 1)
        elif action == 2:  # Right
            col = min(col + 1, self.ncol - 1)
        elif action == 3:  # Up
            row = max(row - 1, 0)
        return row, col

    def _build_transitions(self):
        """
        Build the full transition probability table.
        P[s][a] = [(probability, next_state, reward, terminated), ...]
        """
        P = defaultdict(lambda: defaultdict(list))

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self._to_state(row, col)
                tile = self.desc[row, col]

                for a in range(4):
                    if tile in ('H', 'G'):
                        # Terminal states: agent stays in place, episode is over
                        P[s][a] = [(1.0, s, 0.0, True)]
                    else:
                        if self.is_slippery:
                            # Slippery: intended direction + 2 perpendicular directions
                            # Each with probability 1/3
                            for slip_action in [(a - 1) % 4, a, (a + 1) % 4]:
                                nr, nc = self._move(row, col, slip_action)
                                ns = self._to_state(nr, nc)
                                next_tile = self.desc[nr, nc]
                                reward = 1.0 if next_tile == 'G' else 0.0
                                done = next_tile in ('H', 'G')
                                P[s][a].append((1.0 / 3.0, ns, reward, done))
                        else:
                            # Deterministic: move exactly in chosen direction
                            nr, nc = self._move(row, col, a)
                            ns = self._to_state(nr, nc)
                            next_tile = self.desc[nr, nc]
                            reward = 1.0 if next_tile == 'G' else 0.0
                            done = next_tile in ('H', 'G')
                            P[s][a] = [(1.0, ns, reward, done)]

        return P

    def reset(self, seed=None):
        """Reset environment to start state. Returns (state, info_dict)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.state = self.start_state
        return self.state, {}

    def step(self, action):
        """
        Take an action. Returns (next_state, reward, terminated, truncated, info).
        The transition is sampled from P[state][action] according to probabilities.
        """
        transitions = self.P[self.state][action]
        probs = [t[0] for t in transitions]
        idx = self._rng.choice(len(transitions), p=probs)
        prob, next_state, reward, done = transitions[idx]
        self.state = next_state
        return next_state, reward, done, False, {}

    def close(self):
        pass


# =============================================================================
# 2. CLIFF WALKING
# =============================================================================
class CliffWalking:
    """
    CliffWalking-v0 environment.

    4×12 gridworld. The agent starts at the bottom-left corner (3, 0) and
    must reach the bottom-right corner (3, 11) — the goal.

    The bottom row between start and goal (columns 1–10) is the "cliff."
    Stepping on the cliff gives a reward of -100 and teleports the agent
    back to the start. All other moves give a reward of -1.

    This environment is DETERMINISTIC — chosen actions always succeed.

    This is the classic environment for demonstrating on-policy vs off-policy:
        - Q-LEARNING (off-policy) learns the OPTIMAL path along the cliff edge
          (bottom row), because it always bootstraps from the max Q-value.
          But during training with ε-greedy exploration, the agent frequently
          slips into the cliff, causing poor training rewards.
        - SARSA (on-policy) learns the SAFE path along the top of the grid,
          because it accounts for the exploration noise in its updates.
          This yields better training rewards but a suboptimal final policy.

    Actions: 0=Up, 1=Right, 2=Down, 3=Left

    Grid layout:
        Row 0: . . . . . . . . . . . .
        Row 1: . . . . . . . . . . . .
        Row 2: . . . . . . . . . . . .
        Row 3: S C C C C C C C C C C G

        S = Start, G = Goal, C = Cliff

    State encoding: state = row * 12 + col (0 to 47)
    """

    def __init__(self):
        self.nrow = 4
        self.ncol = 12

        n_states = self.nrow * self.ncol  # 48 states
        n_actions = 4

        self.observation_space = DiscreteSpace(n_states)
        self.action_space = DiscreteSpace(n_actions)

        self.start_state = self._to_state(3, 0)
        self.goal_state = self._to_state(3, 11)

        # Cliff positions: bottom row, columns 1 through 10
        self.cliff = set()
        for c in range(1, 11):
            self.cliff.add(self._to_state(3, c))

        self.P = self._build_transitions()
        self.state = self.start_state
        self._rng = np.random.default_rng()

    def _to_state(self, row, col):
        return row * self.ncol + col

    def _from_state(self, s):
        return s // self.ncol, s % self.ncol

    def _build_transitions(self):
        P = defaultdict(lambda: defaultdict(list))

        # Action deltas: 0=Up, 1=Right, 2=Down, 3=Left
        deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self._to_state(row, col)

                for a in range(4):
                    if s == self.goal_state:
                        # Goal is terminal
                        P[s][a] = [(1.0, s, 0.0, True)]
                        continue

                    dr, dc = deltas[a]
                    nr = max(0, min(self.nrow - 1, row + dr))
                    nc = max(0, min(self.ncol - 1, col + dc))
                    ns = self._to_state(nr, nc)

                    if ns in self.cliff:
                        # Fell off cliff: -100 reward, teleport back to start
                        P[s][a] = [(1.0, self.start_state, -100.0, False)]
                    elif ns == self.goal_state:
                        # Reached goal
                        P[s][a] = [(1.0, ns, -1.0, True)]
                    else:
                        # Normal step
                        P[s][a] = [(1.0, ns, -1.0, False)]

        return P

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.state = self.start_state
        return self.state, {}

    def step(self, action):
        transitions = self.P[self.state][action]
        # Deterministic environment: always one transition with prob 1.0
        prob, next_state, reward, done = transitions[0]
        self.state = next_state
        return next_state, reward, done, False, {}

    def close(self):
        pass


# =============================================================================
# 3. TAXI
# =============================================================================
class Taxi:
    """
    Taxi-v3 environment.

    5×5 grid with 4 designated pickup/dropoff locations marked R, G, Y, B.
    A taxi must navigate to the passenger, pick them up, navigate to the
    destination, and drop them off.

    State encoding (500 total states):
        state = ((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + dest_idx
        - taxi_row, taxi_col: position of the taxi (0–4 each)
        - passenger_loc: 0–3 = at one of 4 locations, 4 = in the taxi
        - dest_idx: 0–3 = which of 4 locations is the destination

    Actions (6 total):
        0 = South (move down)
        1 = North (move up)
        2 = East  (move right)
        3 = West  (move left)
        4 = Pickup
        5 = Dropoff

    Rewards:
        +20  for successful delivery (dropoff at correct destination)
        -10  for illegal pickup (no passenger at location) or illegal dropoff
        -1   for each step (encourages efficiency)

    The grid has WALLS that block east–west movement in certain cells.
    The map looks like:

        +---------+
        |R: | : :G|     Locations: R=(0,0), G=(0,4), Y=(4,0), B=(4,3)
        | : | : : |     Walls shown as '|' block movement between columns
        | : : : : |
        | | : | : |
        |Y| : |B: |
        +---------+

    This environment tests SAMPLE EFFICIENCY — with 500 states × 6 actions
    = 3000 Q-values to learn, algorithms need many episodes to converge.
    """

    def __init__(self):
        self.nrow = 5
        self.ncol = 5

        # The 4 named locations: R, G, Y, B
        self.locs = [(0, 0), (0, 4), (4, 0), (4, 3)]

        # Walls: stored as set of (row, left_col) meaning there's a wall
        # between left_col and left_col+1 at this row
        # From the Taxi map:
        #   Rows 0–1: wall between col 1 and col 2
        #   Rows 3–4: wall between col 0 and col 1
        #   Rows 3–4: wall between col 2 and col 3
        self.walls = set()
        for r in [0, 1]:
            self.walls.add((r, 1))   # wall between col 1 and col 2
        for r in [3, 4]:
            self.walls.add((r, 0))   # wall between col 0 and col 1
            self.walls.add((r, 2))   # wall between col 2 and col 3

        n_states = 500   # 5 × 5 × 5 × 4
        n_actions = 6

        self.observation_space = DiscreteSpace(n_states)
        self.action_space = DiscreteSpace(n_actions)

        self.P = self._build_transitions()
        self.state = None
        self._rng = np.random.default_rng()

    def _encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        """Encode the 4-tuple state into a single integer."""
        return ((taxi_row * 5 + taxi_col) * 5 + pass_loc) * 4 + dest_idx

    def _decode(self, state):
        """Decode a single integer state back into the 4-tuple."""
        dest_idx = state % 4
        state //= 4
        pass_loc = state % 5
        state //= 5
        taxi_col = state % 5
        taxi_row = state // 5
        return taxi_row, taxi_col, pass_loc, dest_idx

    def _wall_between(self, row, col1, col2):
        """Check if there's a wall between (row, col1) and (row, col2)."""
        if col2 == col1 + 1:
            return (row, col1) in self.walls
        elif col2 == col1 - 1:
            return (row, col2) in self.walls
        return False

    def _build_transitions(self):
        """Build the complete transition model for all 500 × 6 = 3000 transitions."""
        P = defaultdict(lambda: defaultdict(list))

        for taxi_row in range(5):
            for taxi_col in range(5):
                for pass_loc in range(5):
                    for dest_idx in range(4):
                        s = self._encode(taxi_row, taxi_col, pass_loc, dest_idx)

                        for action in range(6):
                            new_row, new_col = taxi_row, taxi_col
                            new_pass = pass_loc
                            reward = -1.0
                            done = False

                            if action == 0:    # South
                                new_row = min(taxi_row + 1, 4)
                            elif action == 1:  # North
                                new_row = max(taxi_row - 1, 0)
                            elif action == 2:  # East
                                if taxi_col < 4 and not self._wall_between(taxi_row, taxi_col, taxi_col + 1):
                                    new_col = taxi_col + 1
                            elif action == 3:  # West
                                if taxi_col > 0 and not self._wall_between(taxi_row, taxi_col, taxi_col - 1):
                                    new_col = taxi_col - 1
                            elif action == 4:  # Pickup
                                if pass_loc < 4 and (taxi_row, taxi_col) == self.locs[pass_loc]:
                                    new_pass = 4  # Passenger is now in the taxi
                                else:
                                    reward = -10.0  # Illegal pickup attempt
                            elif action == 5:  # Dropoff
                                if pass_loc == 4:  # Passenger is in taxi
                                    if (taxi_row, taxi_col) == self.locs[dest_idx]:
                                        # Correct destination — success!
                                        new_pass = dest_idx
                                        reward = 20.0
                                        done = True
                                    else:
                                        reward = -10.0  # Wrong dropoff location
                                else:
                                    reward = -10.0  # No passenger in taxi

                            ns = self._encode(new_row, new_col, new_pass, dest_idx)
                            P[s][action] = [(1.0, ns, reward, done)]

        return P

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Random initial configuration
        taxi_row = int(self._rng.integers(0, 5))
        taxi_col = int(self._rng.integers(0, 5))
        pass_loc = int(self._rng.integers(0, 4))    # At one of 4 locations
        dest_idx = int(self._rng.integers(0, 4))

        # Ensure passenger location and destination are different
        while dest_idx == pass_loc:
            dest_idx = int(self._rng.integers(0, 4))

        self.state = self._encode(taxi_row, taxi_col, pass_loc, dest_idx)
        return self.state, {}

    def step(self, action):
        transitions = self.P[self.state][action]
        # Deterministic: single transition with probability 1.0
        prob, next_state, reward, done = transitions[0]
        self.state = next_state
        return next_state, reward, done, False, {}

    def close(self):
        pass


# =============================================================================
# FACTORY FUNCTION — mimics gym.make()
# =============================================================================
def make_env(env_id, **kwargs):
    """
    Create an environment by ID, matching gymnasium.make() interface.

    Usage:
        env = make_env("FrozenLake-v1", map_name="4x4", is_slippery=True)
        env = make_env("CliffWalking-v0")
        env = make_env("Taxi-v3")
    """
    if env_id == "FrozenLake-v1":
        return FrozenLake(
            map_name=kwargs.get("map_name", "4x4"),
            is_slippery=kwargs.get("is_slippery", True)
        )
    elif env_id == "CliffWalking-v0":
        return CliffWalking()
    elif env_id == "Taxi-v3":
        return Taxi()
    else:
        raise ValueError(f"Unknown environment: {env_id}")


# =============================================================================
# QUICK TEST — run this file directly to verify environments work
# =============================================================================
if __name__ == "__main__":
    print("Testing environments...\n")

    # Test FrozenLake 4x4
    env = make_env("FrozenLake-v1", map_name="4x4", is_slippery=True)
    state, _ = env.reset(seed=42)
    print(f"FrozenLake-4x4: {env.observation_space.n} states, {env.action_space.n} actions, start={state}")
    ns, r, done, _, _ = env.step(2)  # Move right
    print(f"  Step(Right): next_state={ns}, reward={r}, done={done}")
    print(f"  P[0][2] (transitions from state 0, action Right): {env.P[0][2]}")

    # Test FrozenLake 8x8
    env = make_env("FrozenLake-v1", map_name="8x8", is_slippery=True)
    state, _ = env.reset(seed=42)
    print(f"\nFrozenLake-8x8: {env.observation_space.n} states, {env.action_space.n} actions, start={state}")

    # Test CliffWalking
    env = make_env("CliffWalking-v0")
    state, _ = env.reset(seed=42)
    print(f"\nCliffWalking: {env.observation_space.n} states, {env.action_space.n} actions, start={state}")
    ns, r, done, _, _ = env.step(1)  # Move right
    print(f"  Step(Right): next_state={ns}, reward={r}, done={done}")
    # Step into cliff
    env.state = env._to_state(2, 5)  # Row 2, col 5
    ns, r, done, _, _ = env.step(2)  # Move down into cliff
    print(f"  Step into cliff from (2,5): next_state={ns}, reward={r}, done={done}")

    # Test Taxi
    env = make_env("Taxi-v3")
    state, _ = env.reset(seed=42)
    tr, tc, pl, di = env._decode(state)
    print(f"\nTaxi: {env.observation_space.n} states, {env.action_space.n} actions")
    print(f"  Initial: taxi=({tr},{tc}), passenger_at={env.locs[pl]}, dest={env.locs[di]}")
    ns, r, done, _, _ = env.step(4)  # Try illegal pickup
    print(f"  Illegal pickup: reward={r}")

    print("\nAll environments working!")
