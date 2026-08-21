# Reinforcement Learning in GridWorld

A custom Reinforcement Learning project that implements and compares three fundamental RL approaches in an 8×8 GridWorld environment:

* Value Iteration
* Q-Learning
* REINFORCE Policy Gradient

The project was developed from scratch to explore how model-based, value-based, and policy-based reinforcement learning methods behave under the same environment and reward structure.

## Project Overview

The environment is a custom deterministic 8×8 GridWorld containing:

* Static obstacles that block movement
* Collectible items that increase the agent's reward
* A dynamic moving obstacle that applies a large collision penalty
* A terminal goal state
* A step penalty that encourages shorter and more efficient paths

The main objective is to evaluate different Reinforcement Learning paradigms on the same task and compare their convergence behavior, training speed, sample efficiency, and memory requirements.

## State Representation

To preserve the Markov property, each state contains all the information required to describe the current environment:

```text
State = (row, column, collected_mask, obstacle_phase)
```

The state space consists of:

```text
54 valid grid cells
× 8 collectible combinations
× 4 obstacle phases
= 1,728 states
```

## Reward Function

The reward structure is designed to encourage successful and efficient navigation.

| Event                      | Reward |
| -------------------------- | -----: |
| Reach Goal                 |   +100 |
| Obtain Collectible         |    +10 |
| Dynamic Obstacle Collision |    -50 |
| Standard Step              |     -1 |

The positive rewards encourage task completion and exploration, while collision and step penalties discourage unsafe behavior and unnecessarily long trajectories.

## Implemented Algorithms

### 1. Value Iteration

A model-based Dynamic Programming approach using the Bellman Optimality Equation.

```text
V(s) = max_a Σ P(s' | s,a) [R + γV(s')]
```

Value Iteration assumes complete knowledge of the environment transition model and provides a mathematical baseline for the optimal policy in the deterministic GridWorld.

### 2. Q-Learning

A model-free Temporal Difference learning algorithm.

The agent maintains a Q-table containing an estimated value for every state-action pair and improves its behavior directly through interaction with the environment.

The implementation uses exploration with epsilon decay to gradually transition from exploration toward exploitation.

### 3. REINFORCE

A Monte Carlo Policy Gradient algorithm that directly optimizes a stochastic policy.

The implementation:

* Uses a Softmax policy for action selection
* Collects complete trajectories
* Computes cumulative returns
* Updates policy parameters after each episode
* Uses gradient ascent to maximize expected reward

## Training Pipeline

The general training workflow is:

```text
Initialize
   ↓
Interact with Environment
   ↓
Collect State / Reward / Termination Data
   ↓
Update Values or Policy
   ↓
Repeat Until Convergence
```

This common pipeline makes it possible to evaluate the algorithms consistently under the same environment conditions.

## Project Structure

```text
.
├── environment.py
├── value_iteration.py
├── q_learning.py
├── reinforce.py
├── visualize.py
└── train_and_compare.py
```

### `environment.py`

Contains the main GridWorld mechanics, including:

* State transitions
* Static obstacles
* Moving obstacle behavior
* Reward logic
* Terminal states

### `value_iteration.py`

Implements the Dynamic Programming solver using full knowledge of the environment transition model.

### `q_learning.py`

Implements the tabular Q-Learning agent with epsilon-decay exploration.

### `reinforce.py`

Implements the stochastic REINFORCE policy-gradient agent using Softmax action selection.

### `visualize.py`

Provides Pygame/Matplotlib visualization for observing the agent's navigation and behavior.

### `train_and_compare.py`

Runs the algorithms and generates unified evaluation metrics and comparison plots.

## Experimental Comparison

The three approaches show different strengths and weaknesses:

| Metric            | Value Iteration               | Q-Learning            | REINFORCE                   |
| ----------------- | ----------------------------- | --------------------- | --------------------------- |
| Training Speed    | Fast                          | Medium                | Slow                        |
| Convergence       | Strong deterministic baseline | Asymptotic            | Higher variance             |
| Sample Efficiency | Model-based                   | High                  | Lower                       |
| Memory            | Moderate                      | Higher due to Q-table | Low policy-weight footprint |

For this discrete environment, Q-Learning provides a strong balance between computational efficiency, model-free learning, and final task performance. Value Iteration provides the optimal mathematical baseline, while REINFORCE demonstrates direct policy optimization but with higher variance and slower learning.

## Results

The training results show that Q-Learning steadily approaches the Value Iteration baseline, while REINFORCE improves more slowly and exhibits noticeably higher variance during training.

The comparison highlights the practical differences between value-based and policy-based learning methods when applied to a relatively small discrete state space.

## Project Strengths

The project focuses on three main strengths:

* Fair comparison using the same environment for all algorithms
* Clean implementations developed without relying on black-box RL frameworks
* Visualization tools for inspecting agent movement, path selection, and collision avoidance

## Limitations

Current limitations include:

* Q-Learning relies on a discrete tabular state representation
* The approach does not scale efficiently to extremely large state spaces
* Deep Q-Networks were not implemented
* Environment transitions are deterministic and do not currently include stochastic effects such as wind or slippery tiles

## Future Work

Possible extensions include:

* Deep Q-Networks (DQN)
* Actor-Critic methods such as A2C
* Proximal Policy Optimization (PPO)
* Stochastic environment transitions
* Larger or continuous state spaces
* Multi-agent Reinforcement Learning

## Key Takeaway

This project demonstrates how three different Reinforcement Learning paradigms solve the same sequential decision-making problem.

For the implemented GridWorld, Q-Learning proved to be the most effective model-free approach, while Value Iteration provided a strong mathematical baseline. The project also connects theoretical concepts such as Bellman equations, Temporal Difference learning, and Policy Gradients with practical implementation.

## Authors

* Ahmed Mamdouh
* Rewan Hany

## Course

Advanced Machine Learning

## License

This project is intended for educational and academic purposes.
