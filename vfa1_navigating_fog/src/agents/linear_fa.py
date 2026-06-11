"""Linear Function Approximation Q-Learning agent for VFA-1.

Approximates Q-values with a linear model:

    Q(s, a) ≈ phi(s, a)^T · theta

where ``phi`` is the hand-crafted 5-dimensional feature vector defined in
``src.features.feature_extractor`` and ``theta`` is a shared weight vector
learned via the semi-gradient TD(0) update:

    delta = r + gamma * max_a' Q(s', a') − Q(s, a)
    theta  ← theta + alpha * delta * phi(s, a)

Because ``phi`` can be computed for *any* observation, including states never
seen during training, the agent generalises across grid sizes and random map
layouts where the tabular agent fails.
"""

from __future__ import annotations

import numpy as np

from src.features.feature_extractor import N_FEATURES, phi


class LinearFAAgent:
    """Linear Function Approximation agent with epsilon-greedy exploration.

    A single weight vector ``theta`` is shared across all actions; only
    ``phi(s, a)`` changes with the action (feature 0 flags a wall directly
    ahead in direction *a*).  Generalisation comes from the fact that
    ``phi`` is defined everywhere, not just on visited states.

    Args:
        n_actions:      Size of the discrete action space (4 for FogGridWorld).
        max_steps:      Episode step limit; must match ``FogGridWorld.max_steps``.
                        Used to normalise the ``step_progress`` feature in ``phi``.
        alpha:          Learning rate for the semi-gradient update.
        gamma:          Discount factor — weight of future rewards.
        epsilon_start:  Initial exploration probability (1.0 = fully random).
        epsilon_end:    Floor for epsilon; exploration never drops below this.
        epsilon_decay:  Multiplicative factor applied to epsilon each episode.
    """

    def __init__(
        self,
        n_actions: int,
        max_steps: int = 200,
        alpha: float = 0.01,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        """Initialise agent; theta starts at zeros, epsilon at epsilon_start."""
        self.n_actions = n_actions
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Shared weight vector; length equals number of features in phi(s,a)
        self.theta: np.ndarray = np.zeros(N_FEATURES, dtype=np.float64)

        # Steps taken so far in the current episode — used for step_progress
        self._steps: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Choose an action using the epsilon-greedy policy.

        During training (``greedy=False``) the agent explores with probability
        ``epsilon`` and exploits the current linear model otherwise.  Pass
        ``greedy=True`` at evaluation time for pure exploitation.

        Args:
            obs:    Current observation from the environment (float32 array).
            greedy: When True, always pick the argmax action.

        Returns:
            Integer action in ``[0, n_actions)``.
        """
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))

        q_values = [self._q(obs, a, self._steps) for a in range(self.n_actions)]
        return int(np.argmax(q_values))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool = False,
    ) -> None:
        """Apply the semi-gradient TD(0) update to theta.

        Update rule::

            delta = r + gamma * max_a' Q(s', a') − Q(s, a)
            theta  ← theta + alpha * delta * phi(s, a)

        On terminal transitions the bootstrap term is dropped and the target
        is just ``r``.  ``step_progress`` for the successor state is evaluated
        at ``self._steps + 1`` so the feature is temporally consistent.

        Args:
            obs:        Observation before the step (s).
            action:     Action taken (a).
            reward:     Scalar reward received (r).
            next_obs:   Observation after the step (s').
            terminated: True if the episode ended naturally (goal/trap/energy=0).
            truncated:  True if the episode was cut off by the step limit.
        """
        step = self._steps
        features = phi(obs, action, step, self.max_steps)
        q_sa = float(features @ self.theta)

        if terminated:
            target = reward
        else:
            next_step = step + 1
            next_q_max = max(
                self._q(next_obs, a, next_step) for a in range(self.n_actions)
            )
            target = reward + self.gamma * next_q_max

        delta = target - q_sa
        self.theta += self.alpha * delta * features

        self._steps = 0 if (terminated or truncated) else step + 1

    def decay_epsilon(self) -> None:
        """Decay exploration rate by one multiplicative step.

        Call once at the end of each training episode.  Clamped at
        ``epsilon_end`` so the agent never stops exploring entirely.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _q(self, obs: np.ndarray, action: int, step: int) -> float:
        """Compute Q(s, a) ≈ phi(s, a)^T · theta at a given episode step.

        Args:
            obs:    Observation vector.
            action: Integer action.
            step:   Step count within the episode (for step_progress feature).

        Returns:
            Scalar Q-value estimate.
        """
        return float(phi(obs, action, step, self.max_steps) @ self.theta)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def weights(self) -> np.ndarray:
        """Current weight vector theta (copy).

        Returns:
            Float64 array of shape (N_FEATURES,) with one weight per feature.
        """
        return self.theta.copy()
