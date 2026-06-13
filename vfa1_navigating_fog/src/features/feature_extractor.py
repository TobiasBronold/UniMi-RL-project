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

# The 7x7 view window (r_max=3) is stored row-major at obs[0..48].
# Centre cell (agent position): row 3, col 3 → flat index 24.
# Maps each action to the flat index of the immediately adjacent cell.
_ADJACENT_IDX: dict[int, int] = {
    Action.UP:    17,  # (dr=-1, dc= 0) → row 2, col 3  → 2*7+3 = 17
    Action.DOWN:  31,  # (dr=+1, dc= 0) → row 4, col 3  → 4*7+3 = 31
    Action.LEFT:  23,  # (dr= 0, dc=-1) → row 3, col 2  → 3*7+2 = 23
    Action.RIGHT: 25,  # (dr= 0, dc=+1) → row 3, col 4  → 3*7+4 = 25
}

_VIEW_SIDE: int = 7    # 2*r_max+1 = 7 for r_max=3
_N_CELLS: int = 49     # _VIEW_SIDE ** 2


# ---------------------------------------------------------------------------
# SimpleFeatureExtractor
# ---------------------------------------------------------------------------

def _action_aligns_with_goal(cells: np.ndarray, action: int) -> float:
    """Return 1.0 if *action* moves the agent closer to the visible goal.

    Computes the signed row/col offset from the agent (centre of the 5×5
    window) to the goal cell, then checks whether the given action reduces
    that distance.  Returns 0.0 when no goal is visible.
    """
    goal_idx = np.where(cells == CellType.GOAL)[0]
    if len(goal_idx) == 0:
        return 0.0
    i = int(goal_idx[0])
    dr = i // _VIEW_SIDE - 3   # positive = goal is below agent
    dc = i  % _VIEW_SIDE - 3   # positive = goal is to the right
    if action == Action.UP    and dr < 0: return 1.0
    if action == Action.DOWN  and dr > 0: return 1.0
    if action == Action.LEFT  and dc < 0: return 1.0
    if action == Action.RIGHT and dc > 0: return 1.0
    return 0.0


def _action_aligns_with_energy(cells: np.ndarray, action: int) -> float:
    """Return 1.0 if *action* moves the agent toward the nearest energy pickup."""
    energy_idx = np.where(cells == CellType.ENERGY)[0]
    if len(energy_idx) == 0:
        return 0.0
    nearest = int(min(energy_idx,
                      key=lambda i: abs(i // _VIEW_SIDE - 3) + abs(i % _VIEW_SIDE - 3)))
    dr = nearest // _VIEW_SIDE - 3
    dc = nearest  % _VIEW_SIDE - 3
    if action == Action.UP    and dr < 0: return 1.0
    if action == Action.DOWN  and dr > 0: return 1.0
    if action == Action.LEFT  and dc < 0: return 1.0
    if action == Action.RIGHT and dc > 0: return 1.0
    return 0.0


class SimpleFeatureExtractor:
    """Five-feature extractor for Linear FA.

    All features are bounded in [0, 1] for numerical stability.

    ======  ========================  ==============================================
    Index   Name                      Description
    ======  ========================  ==============================================
    0       wall_ahead                1.0 if the cell in direction *action* is WALL.
    1       trap_ahead                1.0 if the cell in direction *action* is TRAP.
    2       moving_toward_goal        1.0 if *action* reduces distance to visible goal.
    3       moving_toward_energy      1.0 if *action* reduces distance to nearest pickup.
    4       low_energy                1.0 if energy ≤ 20 % of E_max (urgency flag).
    ======  ========================  ==============================================

    ``energy_level`` (continuous 0→1) was removed: because energy decreases
    monotonically within an episode, every TD update pushed its weight
    negative via bootstrapping — the same spiral that broke ``step_progress``.
    The binary ``low_energy`` flag avoids this: it fires only near depletion
    and does not create a systematic bias in the TD error.
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
            steps:     Steps already taken in the current episode (unused, kept for API).
            max_steps: Maximum steps per episode (unused, kept for API).

        Returns:
            Float64 ndarray of shape ``(5,)``.
        """
        cells = obs[:_N_CELLS]

        wall_ahead           = 1.0 if cells[_ADJACENT_IDX[action]] == CellType.WALL else 0.0
        trap_ahead           = 1.0 if cells[_ADJACENT_IDX[action]] == CellType.TRAP else 0.0
        moving_toward_goal   = _action_aligns_with_goal(cells, action)
        moving_toward_energy = _action_aligns_with_energy(cells, action)
        low_energy = 1.0 if float(obs[_N_CELLS]) < 0.2 else 0.0

        return np.array(
            [wall_ahead, trap_ahead, moving_toward_goal, moving_toward_energy, low_energy],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# RichFeatureExtractor
# ---------------------------------------------------------------------------

class RichFeatureExtractor:
    """Ten-feature extractor for Linear FA.

    Extends SimpleFeatureExtractor with adjacency signals, density, and urgency.
    All features 0–5 are **action-specific** — they differ between actions.

    ======  ========================  ==============================================
    Index   Name                      Description
    ======  ========================  ==============================================
    0       wall_ahead                1.0 if cell in direction *action* is WALL.
    1       trap_ahead                1.0 if cell in direction *action* is TRAP.
    2       moving_toward_goal        1.0 if *action* reduces distance to visible goal.
    3       moving_toward_energy      1.0 if *action* reduces distance to nearest pickup.
    4       goal_adjacent             1.0 if the cell 1 step in *action* direction IS goal.
    5       energy_adjacent           1.0 if the cell 1 step in *action* direction IS energy.
    6       low_energy                1.0 if energy < 30 % of E_max (urgency flag).
    7       walls_norm                Wall count in 5×5 window / 25.
    8       free_cells_norm           FREE cell count in 5×5 window / 25.
    9       rho_norm                  Current visible radius / r_max ∈ [r_min/r_max, 1].
    ======  ========================  ==============================================

    Args:
        r_max: Maximum visible radius; must match ``FogGridWorld.r_max``.
        r_min: Minimum visible radius; must match ``FogGridWorld.r_min``.
    """

    #: Number of features returned by :meth:`phi`.
    n_features: int = 10

    def __init__(self, r_max: int = 3, r_min: int = 1) -> None:
        self.r_max = r_max
        self.r_min = r_min

    def phi(
        self,
        obs: np.ndarray,
        action: int,
        steps: int,
        max_steps: int,
    ) -> np.ndarray:
        """Compute the 11-element feature vector for ``(obs, action)``.

        Args:
            obs:       Float32 observation of length 26 from FogGridWorld.
            action:    Integer action in {0=UP, 1=DOWN, 2=LEFT, 3=RIGHT}.
            steps:     Steps already taken in the current episode (unused, kept for API).
            max_steps: Maximum steps per episode (unused, kept for API).

        Returns:
            Float64 ndarray of shape ``(10,)``.
        """
        cells = obs[:_N_CELLS]
        energy_norm = float(obs[_N_CELLS])
        adj = cells[_ADJACENT_IDX[action]]

        # ── Features 0–5: action-specific ───────────────────────────────────
        wall_ahead           = 1.0 if adj == CellType.WALL   else 0.0
        trap_ahead           = 1.0 if adj == CellType.TRAP   else 0.0
        moving_toward_goal   = _action_aligns_with_goal(cells, action)
        moving_toward_energy = _action_aligns_with_energy(cells, action)
        goal_adjacent        = 1.0 if adj == CellType.GOAL   else 0.0
        energy_adjacent      = 1.0 if adj == CellType.ENERGY else 0.0

        # ── Feature 6: low-energy urgency (fires at the 3×3 tier ≤ 20 %) ────
        low_energy = 1.0 if energy_norm < 0.2 else 0.0

        # ── Features 7–8: cell-type density ──────────────────────────────────
        walls_norm      = float(np.sum(cells == CellType.WALL)) / _N_CELLS
        free_cells_norm = float(np.sum(cells == CellType.FREE)) / _N_CELLS

        # ── Feature 9: current visible radius (step schedule) ────────────────
        rho = 3 if energy_norm > 0.5 else (2 if energy_norm > 0.2 else 1)
        rho_norm = rho / self.r_max   # ∈ {1/3, 2/3, 1}

        return np.array(
            [
                wall_ahead, trap_ahead, moving_toward_goal, moving_toward_energy,
                goal_adjacent, energy_adjacent,
                low_energy, walls_norm, free_cells_norm, rho_norm,
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
