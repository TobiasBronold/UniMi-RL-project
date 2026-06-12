"""Feature extraction for Linear Function Approximation in VFA-1.

Two extractors are provided:

* **SimpleFeatureExtractor** (5 features) — minimal set covering wall
  proximity, goal/energy visibility, energy level, and episode progress.
* **RichFeatureExtractor** (13 features) — extends Simple with directional
  offsets to goal and nearest energy pickup, cell-count statistics, trap
  visibility, and the current visible radius.

Both expose the same ``phi(obs, action, steps, max_steps) → ndarray`` method
and an ``n_features`` class attribute so ``LinearFAAgent`` can work with
either extractor without modification.

The module-level ``phi()`` function and ``N_FEATURES`` constant are kept for
backward compatibility with existing code and notebooks.
"""

from __future__ import annotations

import numpy as np

from src.environment.cell_types import Action, CellType

# ---------------------------------------------------------------------------
# Shared observation-layout constants
# ---------------------------------------------------------------------------

# The 5x5 view window (r_max=2) is stored row-major at obs[0..24].
# Centre cell (agent position): row 2, col 2 → flat index 12.
# Maps each action to the flat index of the immediately adjacent cell.
_ADJACENT_IDX: dict[int, int] = {
    Action.UP:    7,   # (dr=-1, dc= 0) → row 1, col 2  → 1*5+2 = 7
    Action.DOWN:  17,  # (dr=+1, dc= 0) → row 3, col 2  → 3*5+2 = 17
    Action.LEFT:  11,  # (dr= 0, dc=-1) → row 2, col 1  → 2*5+1 = 11
    Action.RIGHT: 13,  # (dr= 0, dc=+1) → row 2, col 3  → 2*5+3 = 13
}

_VIEW_SIDE: int = 5    # 2*r_max+1 = 5 for r_max=2
_N_CELLS: int = 25     # _VIEW_SIDE ** 2


# ---------------------------------------------------------------------------
# SimpleFeatureExtractor
# ---------------------------------------------------------------------------

class SimpleFeatureExtractor:
    """Five-feature extractor for Linear FA.

    All features are bounded in [0, 1] for numerical stability.

    ======  ====================  ================================================
    Index   Name                  Description
    ======  ====================  ================================================
    0       wall_ahead            1.0 if the cell in direction *action* is WALL.
    1       goal_visible          1.0 if GOAL appears anywhere in the 5×5 window.
    2       energy_level          Normalised energy from obs[25] ∈ [0, 1].
    3       energy_pickup_vis     1.0 if an ENERGY pickup is visible.
    4       step_progress         steps / max_steps; rises 0 → 1 over the episode.
    ======  ====================  ================================================
    """

    #: Number of features returned by :meth:`phi`.
    n_features: int = 5

    def phi(
        self,
        obs: np.ndarray,
        action: int,
        steps: int,
        max_steps: int,
    ) -> np.ndarray:
        """Compute the 5-element feature vector for ``(obs, action)``.

        Args:
            obs:       Float32 observation of length 26 from FogGridWorld.
            action:    Integer action in {0=UP, 1=DOWN, 2=LEFT, 3=RIGHT}.
            steps:     Steps already taken in the current episode.
            max_steps: Maximum steps per episode.

        Returns:
            Float64 ndarray of shape ``(5,)``.
        """
        cells = obs[:_N_CELLS]

        wall_ahead = 1.0 if cells[_ADJACENT_IDX[action]] == CellType.WALL else 0.0
        goal_visible = 1.0 if np.any(cells == CellType.GOAL) else 0.0
        energy_level = float(obs[25])
        energy_pickup_vis = 1.0 if np.any(cells == CellType.ENERGY) else 0.0
        step_progress = steps / max_steps if max_steps > 0 else 0.0

        return np.array(
            [wall_ahead, goal_visible, energy_level, energy_pickup_vis, step_progress],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# RichFeatureExtractor
# ---------------------------------------------------------------------------

class RichFeatureExtractor:
    """Thirteen-feature extractor for Linear FA.

    Extends SimpleFeatureExtractor with directional and structural features.
    Features 0–4 are identical to SimpleFeatureExtractor.  Features 5–12
    provide richer spatial context about goal direction, wall density, and
    the current visible radius.

    Delta features (5, 6, 10, 11) are in [−1, 1]; all others in [0, 1].

    ======  =====================  ==============================================
    Index   Name                   Description
    ======  =====================  ==============================================
    0       wall_ahead             1.0 if cell in direction *action* is WALL.
    1       goal_visible           1.0 if GOAL is anywhere in the 5×5 window.
    2       energy_level           Normalised energy ∈ [0, 1].
    3       energy_pickup_vis      1.0 if ENERGY pickup is visible.
    4       step_progress          steps / max_steps.
    5       delta_row_goal         (goal_row − 2) / r_max; 0 if goal not visible.
                                   Negative = goal is above the agent.
    6       delta_col_goal         (goal_col − 2) / r_max; 0 if goal not visible.
                                   Negative = goal is to the left.
    7       walls_norm             Wall count in 5×5 window / 25.
    8       free_cells_norm        FREE cell count in 5×5 window / 25.
    9       trap_visible           1.0 if TRAP is visible anywhere.
    10      delta_row_energy       Row offset to nearest ENERGY / r_max; 0 if none.
    11      delta_col_energy       Col offset to nearest ENERGY / r_max; 0 if none.
    12      rho_norm               Current visible radius / r_max ∈ [r_min/r_max, 1].
    ======  =====================  ==============================================

    Args:
        r_max: Maximum visible radius; must match ``FogGridWorld.r_max``.
        r_min: Minimum visible radius; must match ``FogGridWorld.r_min``.
    """

    #: Number of features returned by :meth:`phi`.
    n_features: int = 13

    def __init__(self, r_max: int = 2, r_min: int = 1) -> None:
        """Store radii used for normalization and rho computation."""
        self.r_max = r_max
        self.r_min = r_min

    def phi(
        self,
        obs: np.ndarray,
        action: int,
        steps: int,
        max_steps: int,
    ) -> np.ndarray:
        """Compute the 13-element feature vector for ``(obs, action)``.

        Args:
            obs:       Float32 observation of length 26 from FogGridWorld.
            action:    Integer action in {0=UP, 1=DOWN, 2=LEFT, 3=RIGHT}.
            steps:     Steps already taken in the current episode.
            max_steps: Maximum steps per episode.

        Returns:
            Float64 ndarray of shape ``(13,)``.
        """
        cells = obs[:_N_CELLS]
        energy_norm = float(obs[25])

        # ── Features 0–4 (same as SimpleFeatureExtractor) ───────────────────
        wall_ahead = 1.0 if cells[_ADJACENT_IDX[action]] == CellType.WALL else 0.0
        goal_visible = 1.0 if np.any(cells == CellType.GOAL) else 0.0
        energy_level = energy_norm
        energy_pickup_vis = 1.0 if np.any(cells == CellType.ENERGY) else 0.0
        step_progress = steps / max_steps if max_steps > 0 else 0.0

        # ── Feature 5–6: signed direction to goal ───────────────────────────
        goal_idx = np.where(cells == CellType.GOAL)[0]
        if len(goal_idx) > 0:
            i = int(goal_idx[0])
            delta_row_goal = (i // _VIEW_SIDE - 2) / self.r_max  # ∈ [−1, 1]
            delta_col_goal = (i  % _VIEW_SIDE - 2) / self.r_max  # ∈ [−1, 1]
        else:
            delta_row_goal = 0.0
            delta_col_goal = 0.0

        # ── Feature 7–8: cell-type density ──────────────────────────────────
        walls_norm = float(np.sum(cells == CellType.WALL)) / _N_CELLS
        free_cells_norm = float(np.sum(cells == CellType.FREE)) / _N_CELLS

        # ── Feature 9: trap visibility ───────────────────────────────────────
        trap_visible = 1.0 if np.any(cells == CellType.TRAP) else 0.0

        # ── Feature 10–11: direction to nearest energy pickup ────────────────
        energy_idx = np.where(cells == CellType.ENERGY)[0]
        if len(energy_idx) > 0:
            # Pick the pickup with the smallest Manhattan distance from centre
            nearest = int(
                min(energy_idx, key=lambda i: abs(i // _VIEW_SIDE - 2) + abs(i % _VIEW_SIDE - 2))
            )
            delta_row_energy = (nearest // _VIEW_SIDE - 2) / self.r_max
            delta_col_energy = (nearest  % _VIEW_SIDE - 2) / self.r_max
        else:
            delta_row_energy = 0.0
            delta_col_energy = 0.0

        # ── Feature 12: current visible radius ──────────────────────────────
        rho = max(self.r_min, round(self.r_max * energy_norm))
        rho_norm = rho / self.r_max  # ∈ [r_min/r_max, 1.0]

        return np.array(
            [
                wall_ahead, goal_visible, energy_level,
                energy_pickup_vis, step_progress,
                delta_row_goal, delta_col_goal,
                walls_norm, free_cells_norm,
                trap_visible,
                delta_row_energy, delta_col_energy,
                rho_norm,
            ],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# Backward-compatible module-level API
# ---------------------------------------------------------------------------

#: Feature count for the default (Simple) extractor; kept for existing imports.
N_FEATURES: int = SimpleFeatureExtractor.n_features

_default_extractor: SimpleFeatureExtractor = SimpleFeatureExtractor()


def phi(
    obs: np.ndarray,
    action: int,
    steps: int,
    max_steps: int,
) -> np.ndarray:
    """Module-level wrapper; delegates to ``SimpleFeatureExtractor.phi()``.

    Provided for backward compatibility.  New code should instantiate
    ``SimpleFeatureExtractor`` or ``RichFeatureExtractor`` directly.

    Args:
        obs:       Float32 observation of length 26.
        action:    Integer action in {0, 1, 2, 3}.
        steps:     Steps taken so far in the current episode.
        max_steps: Maximum episode length.

    Returns:
        Float64 ndarray of shape ``(N_FEATURES,)``.
    """
    return _default_extractor.phi(obs, action, steps, max_steps)
