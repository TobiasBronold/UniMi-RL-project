"""Feature extraction for Linear Function Approximation in VFA-1.

Computes phi(s, a): a compact, hand-crafted feature vector that can be
evaluated for *any* observation and action — including states never seen
during training.  This property lets LinearFAAgent generalise across
grid sizes and random map layouts, unlike the tabular agent.

All features are bounded in [0, 1] for numerical stability.
"""

from __future__ import annotations

import numpy as np

from src.environment.cell_types import Action, CellType

# ---------------------------------------------------------------------------
# Observation layout constants
# ---------------------------------------------------------------------------

# The 5x5 view window (r_max=2) is stored row-major at indices 0..24.
# The agent sits at the centre: row 2, col 2 → flat index 12.
# Each entry maps an action to the flat index of the adjacent cell.
_ADJACENT_IDX: dict[int, int] = {
    Action.UP:    7,   # (dr=-1, dc= 0) → row 1, col 2  → 1*5+2 = 7
    Action.DOWN:  17,  # (dr=+1, dc= 0) → row 3, col 2  → 3*5+2 = 17
    Action.LEFT:  11,  # (dr= 0, dc=-1) → row 2, col 1  → 2*5+1 = 11
    Action.RIGHT: 13,  # (dr= 0, dc=+1) → row 2, col 3  → 2*5+3 = 13
}

#: Number of features in phi(s, a).
N_FEATURES: int = 5


def phi(
    obs: np.ndarray,
    action: int,
    steps: int,
    max_steps: int,
) -> np.ndarray:
    """Compute the feature vector phi(s, a) for a given observation and action.

    Feature index and semantics:

    ======  ====================  ================================================
    Index   Name                  Description
    ======  ====================  ================================================
    0       wall_ahead            1.0 if the cell immediately in direction *action*
                                  is a WALL (or out-of-bounds, encoded as WALL).
    1       goal_visible          1.0 if GOAL appears anywhere in the 5x5 window.
    2       energy_level          Normalised energy, taken directly from obs[-1].
    3       energy_pickup_vis     1.0 if an ENERGY pickup is visible in the window.
    4       step_progress         ``steps / max_steps``; rises from 0 → 1 over the
                                  episode, signalling increasing time pressure.
    ======  ====================  ================================================

    All values are in [0, 1], so the dot product phi^T · theta stays bounded
    even before training has shaped theta.

    Args:
        obs:       Float32 observation vector of length 26 from FogGridWorld.
                   Elements 0..24 are the 5x5 view window; element 25 is the
                   normalised energy level.
        action:    Integer action in {0=UP, 1=DOWN, 2=LEFT, 3=RIGHT}.
        steps:     Number of steps already completed in the current episode.
        max_steps: Maximum steps per episode (matches ``FogGridWorld.max_steps``).

    Returns:
        Float64 ndarray of shape (N_FEATURES,).
    """
    cells = obs[:25]  # 5x5 view window, row-major

    # Feature 0 — wall immediately ahead in the chosen direction
    adj = cells[_ADJACENT_IDX[action]]
    wall_ahead = 1.0 if adj == CellType.WALL else 0.0

    # Feature 1 — goal visible anywhere in the view window
    goal_visible = 1.0 if np.any(cells == CellType.GOAL) else 0.0

    # Feature 2 — normalised energy (already in [0, 1] by the environment)
    energy_level = float(obs[25])

    # Feature 3 — energy pickup visible anywhere in the view window
    energy_pickup_vis = 1.0 if np.any(cells == CellType.ENERGY) else 0.0

    # Feature 4 — fraction of the episode elapsed; urgency signal
    step_progress = steps / max_steps if max_steps > 0 else 0.0

    return np.array(
        [wall_ahead, goal_visible, energy_level, energy_pickup_vis, step_progress],
        dtype=np.float64,
    )
