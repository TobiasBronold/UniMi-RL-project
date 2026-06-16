# VFA-1: Navigating in the Fog

**Course project — Reinforcement Learning, UniMi**

Empirical comparison of **Tabular Q-Learning** vs. **Linear Function Approximation (FA)** in a partially observable gridworld with energy-dependent fog-of-war.

---

## Scientific Objective

Tabular Q-Learning stores one Q-value per (state, action) pair and cannot generalise to unseen states. As the grid grows, the reachable state space explodes — the Q-table fills indefinitely and never converges on random maps.

Linear Function Approximation replaces the table with a compact weight vector `theta` and a hand-crafted feature map `phi(s, a)`. Because features encode *structure* (wall ahead, goal visible, …) rather than raw pixel patterns, the same weights apply to any state — including ones never encountered during training.

The project demonstrates this empirically by training both agent types on random-layout episodes and comparing learning curves, Q-table growth, and goal-reach rates.

---

## Project Structure

```
vfa1_navigating_fog/
├── main.py                          # Entry point — runs a single random-policy episode
├── requirements.txt                 # All Python dependencies (pinned)
│
├── demo/
│   └── demo.ipynb                   # ★ Conference presentation notebook (start here)
│
└── src/
    ├── environment/
    │   ├── grid_world.py            # FogGridWorld — Gymnasium Env subclass
    │   └── cell_types.py            # CellType enum + Action enum
    │
    ├── agents/
    │   ├── tabular_q.py             # Tabular Q-Learning (dictionary Q-table)
    │   └── linear_fa.py             # Linear FA agent (theta · phi)
    │
    ├── features/
    │   └── feature_extractor.py     # SimpleFeatureExtractor (5) & RichFeatureExtractor (10)
    │
    └── experiments/
        ├── saved_agents/            # Pre-trained FA agents (pkl); tabular Q is local-only
        ├── env_visualization.ipynb  # Environment walkthrough
        ├── tabular_q_experiment.ipynb
        ├── linear_fa_experiment.ipynb
        └── fa_comparison_experiment.ipynb
```

---

## MDP Definition

### Observation Space

Fixed-length vector of size **50**:

```
obs = [ cell_0, cell_1, ..., cell_48,  energy_normalized ]
       └─────────── 7×7 view window ──────────────────┘   └── e / E_max ──┘
```

The first 49 entries are a flattened **7×7 window** centred on the agent (`r_max = 3`). The visible radius shrinks in discrete steps as energy depletes:

| Energy level | Visible radius ρ | Window size |
|---|---|---|
| > 50 % | 3 | 7×7 (full sight) |
| 20 %–50 % | 2 | 5×5 |
| ≤ 20 % | 1 | 3×3 (near-blind) |

Cells outside the current radius are filled with `FOG = 0`.

### Cell Encoding

| Value | Meaning |
|-------|---------|
| 0 | FOG — outside visible radius |
| 1 | FREE — walkable cell |
| 2 | WALL — obstacle |
| 3 | TRAP — −1 penalty, episode ends |
| 4 | GOAL — +1 reward, episode ends |
| 5 | ENERGY — energy pickup (+0.2) |

### Action Space

Discrete, 4 actions: `UP=0`, `DOWN=1`, `LEFT=2`, `RIGHT=3`.

### Reward Function

| Event | Reward |
|-------|--------|
| Goal reached | +1.0 |
| Trap entered | −1.0 |
| Energy pickup | +0.2 |
| Wall bump | −0.05 |
| Normal step | −0.01 |

### Episode Structure

- Start: random agent position, random map layout (re-sampled every episode)
- End: goal reached / trap entered / energy = 0 / 200 steps elapsed
- `E_max = 100` (energy budget; depletes 1 per step)

Because the map is re-sampled each episode, memorising a fixed layout is not a viable strategy — the agent must generalise.

---

## Agents

### Agent 1 — Tabular Q-Learning (`src/agents/tabular_q.py`)

Dictionary Q-table keyed on the raw observation tuple. Update rule:

```
Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
```

Parameters: `alpha=0.1`, `gamma=0.99`, epsilon-greedy with decay.

**Limitation:** every distinct observation is a new dictionary key. On random maps the table grows indefinitely and the agent never generalises to unseen states.

### Agent 2 — Simple Linear FA (`src/agents/linear_fa.py` + `SimpleFeatureExtractor`)

Approximates Q-values with a linear model `Q(s,a) ≈ phi(s,a)ᵀ · theta` using **5 binary features**:

| # | Feature | Description |
|---|---------|-------------|
| 0 | `wall_ahead` | 1 if the cell in action direction is a WALL |
| 1 | `trap_ahead` | 1 if the cell in action direction is a TRAP |
| 2 | `moving_toward_goal` | 1 if action reduces Manhattan distance to visible goal |
| 3 | `moving_toward_energy` | 1 if action moves toward nearest visible energy pickup |
| 4 | `low_energy` | 1 if energy ≤ 20 % of E_max |

Parameters: `alpha=0.01`, `gamma=0.99`, epsilon-greedy with decay.

### Agent 3 — Rich Linear FA (`src/agents/linear_fa.py` + `RichFeatureExtractor`)

Extends Simple FA with **10 features** — adds adjacency signals, density statistics, and current visibility radius:

| # | Feature | Description |
|---|---------|-------------|
| 0–4 | *(same as Simple FA)* | |
| 5 | `goal_adjacent` | 1 if the very next cell in action direction is the GOAL |
| 6 | `energy_adjacent` | 1 if the very next cell in action direction is ENERGY |
| 7 | `walls_norm` | Wall density in 7×7 window / 49 |
| 8 | `free_cells_norm` | Free cell density in 7×7 window / 49 |
| 9 | `rho_norm` | Current radius / r_max ∈ {1/3, 2/3, 1} |

---

## Setup

Requires **Python 3.13**. All dependencies are pinned in `requirements.txt`.

```bash
# 1. Create and activate a virtual environment (run from vfa1_navigating_fog/)
python3.13 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify the environment works (random-policy episode)
python main.py
```

---

## Demo Notebook

The main deliverable is the interactive demo notebook:

```bash
cd vfa1_navigating_fog
jupyter notebook demo/demo.ipynb
# then: Kernel → Restart & Run All
```

**The notebook is self-contained.** On first run it automatically trains the Tabular Q agent locally (≈ 2–3 min) if the cached pkl is not present. The two FA agents are loaded from pre-trained weights in `src/experiments/saved_agents/`.

The demo has three sections:

| Section | Content |
|---------|---------|
| **1 — Environment Showcase** | Animated episode of Simple FA on 10×10 and 25×25 grids; fog view, full map, energy timeline |
| **2 — Agent Comparison** | All three agents on the same 25×25 map simultaneously; live step-by-step animation |
| **3 — 500-Episode Statistics** | Reward curves, termination reasons, learned feature weights θ |

> **Note on the Tabular Q agent pkl:** The file is excluded from git because even a 5 000-episode run produces a ~190 MB dictionary (one entry per unique observation). The FA agents (`linear_fa_simple_25x25_35000ep.pkl`, `linear_fa_rich_25x25_35000ep.pkl`) are only ~500 bytes each and are committed directly.

---

## Experiments

The `src/experiments/` folder contains standalone Jupyter notebooks for reproducing individual results:

| Notebook | Content |
|----------|---------|
| `tabular_q_experiment.ipynb` | Tabular Q on random maps; Q-table growth |
| `linear_fa_experiment.ipynb` | Fixed vs. random map comparison for both agent types; learning curves |
| `fa_comparison_experiment.ipynb` | Head-to-head training of all three agents; saves trained weights |
| `env_visualization.ipynb` | Environment walkthrough and cell-type visualisation |
