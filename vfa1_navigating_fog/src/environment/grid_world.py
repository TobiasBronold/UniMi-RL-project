"""Gymnasium environment for the VFA-1 fog-of-war grid world.

A new map is sampled on every reset(), so agents must generalise across
layouts rather than memorise a fixed map.

"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.environment.cell_types import (
    Action,
    CellType,
    REWARD_ENERGY,
    REWARD_GOAL,
    REWARD_STEP,
    REWARD_TRAP,
    REWARD_WALL,
)


# Map each action to its (row_delta, col_delta) displacement.
_ACTION_DELTAS: dict[int, Tuple[int, int]] = {
    Action.UP:    (-1,  0),
    Action.DOWN:  ( 1,  0),
    Action.LEFT:  ( 0, -1),
    Action.RIGHT: ( 0,  1),
}


class FogGridWorld(gym.Env):
    """Partially observable grid world with fog-of-war and energy mechanics.

    Action space — Discrete(4):
        0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

    Args:
        grid_size:    Side length of the square grid.
        r_max:        Maximum visible radius at full energy (5×5 view).
        r_min:        Minimum visible radius at low energy (3×3 view).
        E_max:        Starting energy budget; depletes by 1 each step.
        delta_e:      Energy restored by visiting an ENERGY pickup cell.
        wall_prob:    Per-cell probability of becoming a WALL during map gen.
        trap_count:   Number of TRAP cells placed per episode.
        energy_count: Number of ENERGY pickup cells placed per episode.
        max_steps:    Hard step limit; triggers truncation (not termination).
        render_mode:  Reserved for future use; unused.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        grid_size: int = 10,
        r_max: int = 2,
        r_min: int = 1,
        E_max: int = 100,
        delta_e: int = 30,
        wall_prob: float = 0.15,
        trap_count: int = 3,
        energy_count: int = 3,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.grid_size = grid_size
        self.r_max = r_max
        self.r_min = r_min
        self.E_max = E_max
        self.delta_e = delta_e
        self.wall_prob = wall_prob
        self.trap_count = trap_count
        self.energy_count = energy_count
        self.max_steps = max_steps
        self.render_mode = render_mode

        self._view_side: int = 2 * r_max + 1           # e.g. 5 for r_max=2
        self._obs_size: int = self._view_side ** 2 + 1  # 25 cells + energy = 26

        obs_low = np.zeros(self._obs_size, dtype=np.float32)
        obs_high = np.full(self._obs_size, float(max(CellType)), dtype=np.float32)
        obs_high[-1] = 1.0
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        self.action_space = spaces.Discrete(len(Action))

        # Episode state — all properly initialised in reset()
        self._grid: np.ndarray = np.empty((0, 0), dtype=np.int8)
        self._agent_pos: Tuple[int, int] = (0, 0)
        self._energy: int = 0
        self._steps: int = 0

    # ------------------------------------------------------------------
    # Gymnasium core API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Start a new episode with a freshly generated random map.

        Args:
            seed:    Optional RNG seed for reproducibility.
            options: Unused; accepted for API compatibility.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)  # seeds self.np_random

        self._grid = self._generate_map()
        self._agent_pos = self._sample_free(self._grid)  # type: ignore[assignment]
        if self._agent_pos is None:
            raise RuntimeError("Generated map has no FREE cell for agent placement.")

        self._energy = self.E_max
        self._steps = 0

        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one action and return the next transition.

        Args:
            action: Integer in {0, 1, 2, 3}.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        dr, dc = _ACTION_DELTAS[Action(action)]
        r, c = self._agent_pos
        nr = int(np.clip(r + dr, 0, self.grid_size - 1))
        nc = int(np.clip(c + dc, 0, self.grid_size - 1))

        target = CellType(self._grid[nr, nc])

        self._energy = max(0, self._energy - 1)  # step always costs 1 energy
        self._steps += 1

        terminated = False
        reward = REWARD_STEP

        if target == CellType.WALL:
            # Agent stays; apply wall-bump penalty instead of step penalty
            reward = REWARD_WALL
        else:
            self._agent_pos = (nr, nc)

            if target == CellType.GOAL:
                reward = REWARD_GOAL
                terminated = True
            elif target == CellType.TRAP:
                reward = REWARD_TRAP
                terminated = True
            elif target == CellType.ENERGY:
                reward = REWARD_ENERGY
                self._energy = min(self._energy + self.delta_e, self.E_max)
                self._grid[nr, nc] = CellType.FREE  # consumed after pickup

        if self._energy == 0:
            terminated = True

        # Truncation: time limit exceeded, but no terminal MDP state reached
        truncated = not terminated and (self._steps >= self.max_steps)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Map generation
    # ------------------------------------------------------------------

    def _generate_map(self) -> np.ndarray:
        """Create a random grid layout for the new episode.

        Procedure:
          1. Fill the entire grid with FREE cells.
          2. Sample a wall mask and apply it.
          3. Place special cells (traps, energy pickups, goal) in order
             so the goal is placed last and cannot be overwritten.

        Returns:
            2-D int8 array of shape (grid_size, grid_size).
        """
        grid = np.full(
            (self.grid_size, self.grid_size), int(CellType.FREE), dtype=np.int8
        )

        # Vectorised: sample all wall positions at once
        wall_mask = (
            self.np_random.random((self.grid_size, self.grid_size)) < self.wall_prob
        )
        grid[wall_mask] = int(CellType.WALL)

        for _ in range(self.trap_count):
            pos = self._sample_free(grid)
            if pos is not None:
                grid[pos] = int(CellType.TRAP)

        for _ in range(self.energy_count):
            pos = self._sample_free(grid)
            if pos is not None:
                grid[pos] = int(CellType.ENERGY)

        goal_pos = self._sample_free(grid)
        if goal_pos is not None:
            grid[goal_pos] = int(CellType.GOAL)

        return grid

    def _sample_free(self, grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """Return a uniformly random FREE cell from *grid*, or None.

        Args:
            grid: The grid array to sample from (may differ from self._grid
                  during map generation).

        Returns:
            A (row, col) tuple, or None when no FREE cell exists.
        """
        rows, cols = np.where(grid == int(CellType.FREE))
        if len(rows) == 0:
            return None
        idx = int(self.np_random.integers(len(rows)))
        return int(rows[idx]), int(cols[idx])

    # ------------------------------------------------------------------
    # Observation and info helpers
    # ------------------------------------------------------------------

    def _visible_radius(self) -> int:
        """Compute the current visible radius from remaining energy.

        Formula from DESIGN.md:
            rho(e) = max(r_min, round(r_max * (e / E_max)))

        Returns:
            Integer radius in [r_min, r_max].
        """
        return max(self.r_min, round(self.r_max * (self._energy / self.E_max)))

    def _get_obs(self) -> np.ndarray:
        """Build the fixed-length float32 observation vector.

        Iterates the (2*r_max+1)^2 offsets of the view window, centred on
        the agent, in row-major order (top-left → bottom-right).

        Each position is encoded as:
          CellType.FOG  — if |dr| > rho or |dc| > rho  (outside visible radius)
          CellType.WALL — if the position is outside the grid boundary
          grid value    — otherwise

        Returns:
            Float32 array of length _obs_size = (2*r_max+1)^2 + 1.
        """
        r, c = self._agent_pos
        rho = self._visible_radius()

        cells = np.empty(self._view_side ** 2, dtype=np.float32)
        idx = 0
        for dr in range(-self.r_max, self.r_max + 1):
            for dc in range(-self.r_max, self.r_max + 1):
                if abs(dr) > rho or abs(dc) > rho:
                    cells[idx] = float(CellType.FOG)   # shrouded by fog
                else:
                    gr, gc = r + dr, c + dc
                    if 0 <= gr < self.grid_size and 0 <= gc < self.grid_size:
                        cells[idx] = float(self._grid[gr, gc])
                    else:
                        cells[idx] = float(CellType.WALL)  # grid edge = wall
                idx += 1

        energy_norm = np.float32(self._energy / self.E_max)
        return np.append(cells, energy_norm)

    def _get_info(self) -> dict:
        """Return auxiliary diagnostic information (not part of obs).

        Returns:
            Dict with keys: agent_pos, energy, steps, visible_radius.
        """
        return {
            "agent_pos": self._agent_pos,
            "energy": self._energy,
            "steps": self._steps,
            "visible_radius": self._visible_radius(),
        }
