"""Cell type definitions, action space, and reward constants for VFA-1.

Encoding integers are used both as grid array values and as entries in the 
observation vector, so the numeric assignments must never change once set.
"""

from enum import IntEnum


class CellType(IntEnum):
    """Integer encoding for every possible grid cell type."""

    FOG = 0     # outside the agent's current visible radius (not a real cell)
    FREE = 1    # walkable empty cell
    WALL = 2    # impassable obstacle; agent stays in place on collision
    TRAP = 3    # penalty cell; episode terminates on entry
    GOAL = 4    # target cell; episode terminates on entry
    ENERGY = 5  # pickup cell; replenishes agent energy on entry, then consumed


class Action(IntEnum):
    """Discrete action space — four cardinal directions."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

REWARD_GOAL: float = 1.0    # reaching the goal
REWARD_TRAP: float = -1.0   # entering a trap
REWARD_ENERGY: float = 0.2  # collecting an energy pickup
REWARD_WALL: float = -0.05  # bumping into a wall (no movement)
REWARD_STEP: float = -0.01  # any normal movement step
