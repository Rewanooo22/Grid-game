# GridWorld RL Algorithms

Three tabular reinforcement learning algorithms implemented from scratch with NumPy and compared on the same custom 8x8 GridWorld:

- **Value Iteration** (dynamic programming, uses the environment model)
- **Q-Learning** (temporal-difference control, epsilon-greedy)
- **REINFORCE** (Monte-Carlo policy gradient with a tabular softmax policy)

A single script trains all three, writes comparison plots and a `results.csv` table. Pygame viewers can play back an episode or draw a learned policy as arrows.

> **Note:** the source code is currently packaged as a zip archive, `game (2).zip`, which contains a `game/` folder. Extract it before running anything (see [Getting started](#getting-started)).

## Environment

Defined in `environment.py` as a deterministic, fully known MDP:

- 8x8 grid, start at `(0, 0)`, terminal goal at `(7, 7)`.
- 10 static wall cells; moving into a wall or off the grid leaves the agent in place.
- 3 collectibles at `(0, 7)`, `(4, 0)` and `(7, 3)`, each rewarded the first time it is reached.
- 1 dynamic obstacle that patrols a fixed 4-cell loop along row 2 (columns 4 to 7), advancing one cell per step regardless of the agent's action.
- Actions: up, down, left, right. Episodes end at the goal or after 200 steps.

| Event | Reward |
|---|---:|
| Each step | -1 |
| Collectible (first visit) | +10 |
| Landing on the obstacle's cell | -50 |
| Reaching the goal | +100 |

**State** = `(row, col, collected_mask, obstacle_phase)`. Including the collectible bitmask and the obstacle phase keeps the process Markovian. This gives 54 free cells x 8 masks x 4 phases = **1,728 states**, all enumerated by `GridWorld.enumerate_states()`. The model is exposed through `next_state_reward(s, a)`, with `transition_prob` and `reward_fn` wrappers for the explicit P/R form.

## Algorithms

| | Value Iteration | Q-Learning | REINFORCE |
|---|---|---|---|
| File | `value_iteration.py` | `q_learning.py` | `reinforce.py` |
| Type | Model-based DP | Model-free, off-policy TD | Model-free, Monte-Carlo policy gradient |
| Representation | `V(s)` table | `Q(s, a)` table (1,728 x 4) | Preference table `theta(s, a)` (1,728 x 4), softmax policy |
| Settings in `train_and_compare.py` | gamma 0.99, stop when max delta < 1e-6 | alpha 0.1, gamma 0.99, epsilon decayed linearly 1.0 to 0.05, 5,000 episodes | alpha 0.01, gamma 0.99, 20,000 episodes, running mean-return baseline |

Q-Learning and REINFORCE use a fixed seed (0). All three return a plain `dict` mapping state to action, so any policy can be passed to the visualisation helpers.

## Project structure

Contents of `game (2).zip`:

```text
game/
├── environment.py        # GridWorld MDP: layout, model, rewards, pygame rendering
├── value_iteration.py    # Value Iteration + greedy policy extraction
├── q_learning.py         # Tabular Q-Learning
├── reinforce.py          # Tabular softmax REINFORCE
├── visualize.py          # pygame playback / policy arrows, matplotlib plots
├── train_and_compare.py  # trains all three, writes plots and results.csv
├── requirements.txt      # numpy, matplotlib, pygame
├── results.csv           # results of a previous run
├── reward_comparison.png # plots from that run
├── convergence.png
├── steps_to_goal.png
└── README.md
```

The archive also contains a `.git/` folder with no commits and a `__pycache__/` folder; neither is needed to run the code.

## Getting started

### Prerequisites

- Python 3 (the archived bytecode was produced by Python 3.13)
- A display, if you want to use the pygame viewers

### Installation

```bash
git clone https://github.com/Rewanooo22/gridworld-rl-algorithms.git
cd gridworld-rl-algorithms
unzip "game (2).zip"
cd game
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Windows without `unzip`, extract the archive with Explorer or `tar -xf "game (2).zip"`.

### Train and compare

```bash
python train_and_compare.py
```

This prints progress for each algorithm, prints the results table, and overwrites `results.csv`, `reward_comparison.png`, `convergence.png` and `steps_to_goal.png` in the current folder. The plots are rendered headlessly with matplotlib's Agg backend.

### Watch an agent or draw its policy

The pygame viewers open a window, so call them from a Python shell inside `game/`:

```python
from environment import GridWorld
from value_iteration import value_iteration
from visualize import play_episode, render_policy_arrows

env = GridWorld()
policy, V, log, t = value_iteration(env)

play_episode(env, policy)                                   # animate one episode
render_policy_arrows(env, policy, save_path="arrows.png")   # greedy action per cell for mask=0, phase=0
```

## Results

From `results.csv` included in the archive (one run of `train_and_compare.py`):

| Algorithm | final_avg_reward | steps_to_goal | training_time_sec | convergence_point |
|---|---:|---:|---:|---:|
| Value Iteration | 106.0 | 14 | 0.2161 | 16 |
| Q-Learning | 105.17 | 14 | 3.1622 | 3890 |
| REINFORCE | 82.88 | 27 | 27.5733 | 5275 |

How the columns are computed in `train_and_compare.py`:

- `final_avg_reward`: for Value Iteration, the reward of one greedy rollout from the start state; for Q-Learning and REINFORCE, the mean reward of the last 100 *training* episodes (Q-Learning still explores with epsilon = 0.05 at the end).
- `steps_to_goal`: steps taken by one greedy rollout of the final policy, for all three algorithms.
- `training_time_sec`: wall-clock training time on the machine used for that run.
- `convergence_point`: number of sweeps for Value Iteration; for the other two, the first episode at which the 100-episode moving-average reward comes within 5% of its final level.

In that run, the greedy Q-Learning policy reached the goal in the same 14 steps as the Value Iteration policy, while the tabular REINFORCE policy needed 27 steps and had a lower average training reward. The plots in the archive (`reward_comparison.png`, `convergence.png`, `steps_to_goal.png`) come from the same run.

## Limitations

- Tabular methods only; the state space is small (1,728 states) and fully enumerable.
- Deterministic transitions; there is no stochastic dynamics (e.g. slippery tiles).
- Results come from a single seed and a single run; timings depend on hardware.
- REINFORCE uses only a global running-mean baseline, not a state-dependent value function.
- The code is shipped inside a zip archive rather than as files at the repository root.

## Future improvements (ideas)

- Unpack the source into the repository root and drop the archived `.git/` and `__pycache__/` folders.
- Run multiple seeds and report means and spreads.
- Add an actor-critic baseline or function approximation (e.g. DQN) and stochastic transitions.

## Context and team

Coursework for **Advanced Machine Learning** (AIE322, as referenced in `environment.py`).

Team: Ahmed Mamdouh, Rewan Hany.
