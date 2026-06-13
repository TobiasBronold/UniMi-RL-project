"""Gymnasium environment for the VFA-1 fog-of-war grid world.

A new map is sampled on every reset(), so agents must generalise across
layouts rather than memorise a fixed map.

"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.environment.cell_types import (
    Action,
    CellType,
    REWARD_ENERGY,
    REWARD_GOAL,
    REWARD_STARVE,
    REWARD_STEP,
    REWARD_TRAP,
    REWARD_WALL,
)


# RGB colour palette indexed by CellType value (0–5) plus agent sentinel (6).
# Extend this list when new CellTypes are added.
_PALETTE: list[tuple[int, int, int]] = [
    ( 28,  28,  28),  # 0 FOG    – near-black
    (240, 240, 224),  # 1 FREE   – off-white
    ( 96,  96,  96),  # 2 WALL   – grey
    (204,  51,  51),  # 3 TRAP   – red
    ( 51, 204,  85),  # 4 GOAL   – green
    (255, 170,   0),  # 5 ENERGY – amber
    ( 51, 153, 255),  # 6 AGENT  – blue (sentinel, not a CellType)
]

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
        fixed_map:    If True the map layout is generated once and reused every
                      episode (only the agent start position changes).  Useful
                      for isolating memorisation vs. generalisation.
    """

    metadata: dict = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        grid_size: int = 10,
        r_max: int = 3,
        r_min: int = 1,
        E_max: int = 100,
        delta_e: int = 30,
        wall_prob: float = 0.15,
        trap_count: int = 3,
        energy_count: int = 3,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
        fixed_map: bool = False,
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
        self.fixed_map = fixed_map
        # Stores the canonical layout when fixed_map=True; set on first reset().
        self._fixed_grid: Optional[np.ndarray] = None

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
        """Start a new episode.

        When ``fixed_map=False`` (default) a fresh random map is generated.
        When ``fixed_map=True`` the map layout is generated once on the first
        call and restored from that snapshot on every subsequent reset, so the
        topology never changes between episodes.  Energy pickup cells consumed
        during a prior episode are restored as well.  Only the agent's starting
        position is re-sampled each episode.

        Args:
            seed:    Optional RNG seed for reproducibility.
            options: Unused; accepted for API compatibility.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)  # seeds self.np_random

        if self.fixed_map:
            if self._fixed_grid is None:
                self._fixed_grid = self._generate_map()
            # Restore the canonical layout (energy pickups may have been consumed)
            self._grid = self._fixed_grid.copy()
        else:
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
            reward = REWARD_STARVE
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

        Step schedule (three visibility tiers):
            energy > 50 %  →  r = 3  (7×7 window, full sight)
            energy > 20 %  →  r = 2  (5×5 window)
            energy ≤ 20 %  →  r = 1  (3×3 window, near-blind)

        Returns:
            Integer radius in {1, 2, 3}.
        """
        frac = self._energy / self.E_max
        if frac > 0.5:
            return 3
        elif frac > 0.2:
            return 2
        else:
            return 1

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

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> Optional[np.ndarray]:
        """Return a pixel-art RGB array of the current state.

        Only active when the environment was created with
        ``render_mode="rgb_array"``.  Each grid cell is drawn as a
        ``_CELL_SIZE × _CELL_SIZE`` pixel block; cells outside the
        agent's current visible radius are painted with the FOG colour.

        Returns:
            uint8 array of shape ``(grid_size*32, grid_size*32, 3)``,
            or ``None`` when render_mode is not "rgb_array".
        """
        if self.render_mode != "rgb_array":
            return None

        cs = 32  # pixels per cell
        r_a, c_a = self._agent_pos
        rho = self._visible_radius()

        img = np.zeros((self.grid_size * cs, self.grid_size * cs, 3), dtype=np.uint8)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if r == r_a and c == c_a:
                    color = _PALETTE[6]                       # agent sentinel
                elif abs(r - r_a) > rho or abs(c - c_a) > rho:
                    color = _PALETTE[CellType.FOG]            # outside visible radius
                else:
                    color = _PALETTE[self._grid[r, c]]
                img[r * cs:(r + 1) * cs, c * cs:(c + 1) * cs] = color

        return img

    def render_frame(self, ax: Optional[Any] = None, fog: bool = True) -> Any:
        """Render the current state as a matplotlib Axes for Jupyter notebooks.

        Builds a colour-coded view of the grid.  When ``fog=True``
        (default) cells outside the current visible radius are shown as
        FOG, mirroring the agent's actual observation.  Set
        ``fog=False`` to reveal the full map (useful for debugging).

        Typical notebook usage::

            fig, ax = plt.subplots(figsize=(5, 5))
            env.render_frame(ax)
            plt.tight_layout()
            plt.show()

        For a step-by-step animation use ``IPython.display.clear_output``::

            from IPython.display import clear_output
            obs, _ = env.reset(seed=0)
            while True:
                clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(5, 5))
                env.render_frame(ax)
                plt.tight_layout(); plt.show()
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

        Args:
            ax:  An existing ``matplotlib.axes.Axes`` to draw into, or
                 ``None`` to create a new figure automatically.
            fog: If ``True``, mask cells outside the visible radius with
                 the FOG colour.  If ``False``, show the full grid.

        Returns:
            The populated ``matplotlib.axes.Axes``.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        r_a, c_a = self._agent_pos
        rho = self._visible_radius()

        # Vectorised fog mask: True where cell is outside visible radius
        rows = np.arange(self.grid_size)
        cols = np.arange(self.grid_size)
        fog_mask = (
            (np.abs(rows[:, None] - r_a) > rho) |
            (np.abs(cols[None, :] - c_a) > rho)
        )

        # display uses float so we can store the agent sentinel (6)
        display = self._grid.astype(float)
        if fog:
            display[fog_mask] = float(CellType.FOG)
        display[r_a, c_a] = 6.0  # agent drawn on top of whatever cell it's on

        hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in _PALETTE]
        cmap = ListedColormap(hex_colors)

        ax.imshow(display, cmap=cmap, vmin=0, vmax=6, interpolation="nearest")

        # Minor tick grid lines delineate cells cleanly
        ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.grid(which="minor", color="#222222", linewidth=0.5)
        ax.tick_params(which="both", bottom=False, left=False,
                       labelbottom=False, labelleft=False)

        ax.set_title(
            f"Step {self._steps}  ·  Energy {self._energy}/{self.E_max}"
            f"  ·  ρ = {rho}",
            fontsize=11,
        )

        # Legend: one patch per visible CellType + agent
        legend_handles = [
            mpatches.Patch(
                facecolor=hex_colors[ct.value],
                edgecolor="#444444",
                linewidth=0.5,
                label=ct.name,
            )
            for ct in CellType
            if ct != CellType.FOG
        ] + [
            mpatches.Patch(
                facecolor=hex_colors[6],
                edgecolor="#444444",
                linewidth=0.5,
                label="AGENT",
            )
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8,
            framealpha=0.95,
        )

        return ax

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
