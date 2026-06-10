# VFA-1: Navigating in the Fog

**Course project — Reinforcement Learning, UniMi**

Empirical comparison of **Tabular Q-Learning** vs. **Linear Function Approximation** in a partially observable gridworld with energy-dependent fog-of-war.

---

## Scientific Objective

Tabular Q-Learning stores one Q-value per (state, action) pair and cannot generalise to unseen states. As grid size grows, the observed state space explodes and the agent degrades. Linear Function Approximation replaces the Q-table with a compact weight vector `theta` and a hand-crafted feature map `phi(s, a)`, which applies to any state including unseen ones.

The project shows this degradation empirically across three grid sizes (10×10, 20×20, 50×50) and discusses how the choice of state representation affects both convergence and policy quality.

---

## Project Structure

```
vfa1_navigating_fog/
│
├── main.py                          # Entry point — runs a random-policy episode
├── requirements.txt                 # All Python dependencies (pinned)
│
├── src/
│   ├── __init__.py
│   │
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── grid_world.py            # FogGridWorld — Gymnasium Env subclass
│   │   └── cell_types.py            # CellType enum: FOG=0 FREE=1 WALL=2 TRAP=3 GOAL=4 ENERGY=5
│   │
│   ├── agents/
│   │   ├── tabular_q.py             # Tabular Q-Learning (dictionary-based Q-table)
│   │   └── linear_fa.py             # Linear Function Approximation agent
│   │
│   ├── features/
│   │   └── feature_extractor.py     # phi(s, a) feature vector used by Linear FA
│   │
│   └── experiments/
│       ├── __init__.py
│       ├── runner.py                # Training loop + metric logging
│       └── env_visualization.ipynb  # Jupyter notebook — environment walkthrough & visualisation
│
├── results/                         # Generated plots and CSV logs (created at runtime)
│
└── task/
    ├── DESIGN.md                    # Full MDP specification (authoritative reference)
    └── projectdescription.md        # Original assignment brief from the professor
```

---

## MDP Definition

### State / Observation Space

The agent does not observe the full grid. At each step it receives a fixed-length vector of size **26**:

```
obs = [ cell_0, cell_1, ..., cell_24,  energy_normalized ]
```

The first 25 entries are a flattened 5×5 window centred on the agent. Cells outside the
current visible radius are filled with `FOG = 0`. The visible radius shrinks as energy depletes:

```python
rho(e) = max(r_min, round(r_max * (e / E_max)))
# r_max=2 -> 5x5 view at full energy
# r_min=1 -> 3x3 view at low energy
```

Cell encoding:

| Value | Meaning |
|-------|---------|
| 0 | FOG — outside visible radius |
| 1 | FREE — walkable cell |
| 2 | WALL — obstacle |
| 3 | TRAP — penalty, episode ends |
| 4 | GOAL — target, episode ends |
| 5 | ENERGY — energy pickup |

Key parameters: `r_max=2`, `r_min=1`, `E_max=100`, `obs_size=26`.

### Action Space

Discrete, 4 actions: `UP=0`, `DOWN=1`, `LEFT=2`, `RIGHT=3`.

### Transition Dynamics

Deterministic. Attempting to move into a wall keeps the agent in place (and costs −0.05). Energy decreases by 1 every step; visiting an energy cell partially replenishes it.

### Reward Function

| Event | Reward |
|-------|--------|
| Goal reached | +1.0 |
| Trap entered | −1.0 |
| Energy pickup | +0.2 |
| Wall bump (no movement) | −0.05 |
| Normal step | −0.01 |

### Episode Structure

- Start: random agent position, random map layout (new layout every episode)
- End: goal reached / trap entered / energy = 0 / 200 steps elapsed

Because the map is re-sampled each episode, memorising a fixed layout is not a viable strategy — generalisation is required.

---

## Agents

### Agent 1 — Tabular Q-Learning (`src/agents/tabular_q.py`)

Maintains a Python dictionary `Q[(obs_tuple, action)] -> float`. Update rule:

```
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

Parameters: `alpha=0.1`, `gamma=0.99`, `epsilon=0.1` (epsilon-greedy).

**Expected limitation:** every new grid cell arrangement is a new dictionary key. The agent never generalises — on large grids it will have seen only a tiny fraction of the reachable states.

### Agent 2 — Linear Function Approximation (`src/agents/linear_fa.py`)

Approximates Q-values with a linear model:

```
Q(s,a) ≈ phi(s,a)^T · theta
```

The feature extractor (`src/features/feature_extractor.py`) computes `phi(s, a)`:

| Feature | Description |
|---------|-------------|
| Wall ahead | Is there a wall in direction `a`? |
| Goal visible | Is the goal cell present in the current observation? |
| Energy level | Normalised energy `e / E_max` |
| Distance to goal | Normalised Manhattan distance to goal (0 if not visible) |
| Energy pickup visible | Is an energy cell present in the observation? |

Parameters: `alpha=0.01`, `gamma=0.99`, `epsilon=0.1`.

**Advantage:** `phi(s, a)` can be computed for any state, including ones never seen during training. Generalisation is built into the representation.

---

## Experimental Plan

Both agents are evaluated on three grid sizes:

| Grid size | Tabular Q (expected) | Linear FA (expected) |
|-----------|----------------------|----------------------|
| 10×10 | Converges | Converges |
| 20×20 | Struggles | Converges |
| 50×50 | Breaks down | Converges |

Metrics reported:
- Learning curves (mean episode reward vs. episode number)
- Convergence speed (episodes to reach stable reward)
- Final policy quality (mean reward over last 100 episodes)

---

## Setup & Run

Requires Python 3.13. All dependencies are pinned in `requirements.txt`.

```bash
# create and activate venv (from inside vfa1_navigating_fog/)
python3.13 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Run a single random-policy episode to verify the environment works:

```bash
python main.py
```

Run the full experiment (trains both agents, saves plots to `results/`):

```bash
python -m src.experiments.runner
```

Explore the environment interactively:

```bash
jupyter notebook src/experiments/env_visualization.ipynb
```
