"""Tabular Q-Learning agent for VFA-1: Navigating in the Fog.

The Q-table is a plain Python dictionary keyed on (state_tuple, action) pairs.
Because every distinct observation vector gets its own entry, the table grows
without bound as the grid increases in size — it cannot generalise to states
it has never visited.  That degradation is the scientific focus of VFA-1.

Epsilon-greedy exploration decays multiplicatively after each episode:
    epsilon <- max(epsilon_end, epsilon * epsilon_decay)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np


class TabularQAgent:
    """Dictionary-based Q-Learning agent with decaying epsilon-greedy policy.

    Args:
        n_actions:      Size of the discrete action space (4 for FogGridWorld).
        alpha:          Learning rate — how much each TD error shifts the value.
        gamma:          Discount factor — weight of future rewards.
        epsilon_start:  Initial exploration probability (1.0 = fully random).
        epsilon_end:    Floor for epsilon; exploration never drops below this.
        epsilon_decay:  Multiplicative factor applied to epsilon each episode.
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        """Initialise agent with hyperparameters; Q-table starts empty."""
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # defaultdict(float) returns 0.0 for any unseen (state, action) key,
        # avoiding explicit initialisation and KeyError on first lookup.
        self.Q: defaultdict[tuple, float] = defaultdict(float)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Choose an action with epsilon-greedy policy.

        During training (``greedy=False``) the agent explores with probability
        ``epsilon`` and exploits the current Q-table otherwise.  Pass
        ``greedy=True`` at evaluation time to use the pure greedy policy.

        Args:
            obs:    Current observation from the environment (float32 array).
            greedy: When True, always pick the argmax action (no exploration).

        Returns:
            Integer action in ``[0, n_actions)``.
        """
        if not greedy and np.random.random() < self.epsilon:
            # Random action — uniform over the action space
            return int(np.random.randint(self.n_actions))

        state = self._obs_to_key(obs)
        # Retrieve Q(s,·) for every action; unseen actions default to 0.0
        q_values = [self.Q[(state, a)] for a in range(self.n_actions)]
        # argmax; ties broken by the first maximum (index 0 = UP)
        return int(np.argmax(q_values))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        """Apply the tabular Q-learning update rule.

        Update equation (Sutton & Barto §6.5)::

            Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        On terminal transitions there is no successor state, so the bootstrap
        term ``gamma * max_a' Q(s',a')`` is dropped and the target is just ``r``.

        Args:
            obs:        Observation *before* the step (s).
            action:     Action taken (a).
            reward:     Scalar reward received (r).
            next_obs:   Observation *after* the step (s').
            terminated: True if the episode ended (goal / trap / energy = 0).
                        Distinct from truncation (time limit) — on truncation
                        the bootstrap term is still valid.
        """
        state = self._obs_to_key(obs)
        next_state = self._obs_to_key(next_obs)

        if terminated:
            # No future return on a terminal transition
            target = reward
        else:
            next_q_max = max(self.Q[(next_state, a)] for a in range(self.n_actions))
            target = reward + self.gamma * next_q_max

        # TD error: difference between the target and current estimate
        td_error = target - self.Q[(state, action)]
        self.Q[(state, action)] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        """Decay exploration rate by one multiplicative step.

        Call once at the end of each training episode.  Clamped at
        ``epsilon_end`` so the agent never stops exploring entirely.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs_to_key(self, obs: np.ndarray) -> tuple:
        """Convert a float32 observation array into a hashable tuple.

        ``obs.tolist()`` promotes each numpy float32 to a Python float,
        producing a plain tuple that can be used as a dict key without
        holding a reference to the original numpy array.

        Args:
            obs: Float32 array from FogGridWorld (length 26).

        Returns:
            Tuple of Python floats, one per observation element.
        """
        return tuple(obs.tolist())

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def q_table_size(self) -> int:
        """Number of distinct (state, action) entries in the Q-table.

        A proxy for state-space coverage: grows rapidly on large grids,
        illustrating why tabular methods are memory-inefficient.
        """
        return len(self.Q)
