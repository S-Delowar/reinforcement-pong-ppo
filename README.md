# Reinforcement Learning Pong using PPO
A reinforcement learning project where an AI agent learns to play Atari Pong from raw pixel input using the Proximal Policy Optimization (PPO) algorithm.

This project trains a deep reinforcement learning agent using the PPO algorithm from Stable-Baselines3 to play the Pong game in the Atari Learning Environment. The agent observes pixel-level input, processes it using a convolutional neural network (CNN), and learns an optimal policy through interaction with the environment.

## Key Features
- Raw Pixel Input: Agent learns from high-dimensional observations without any manual feature engineering.

- Frame Stacking: Captures temporal information by stacking consecutive frames as input to the CNN.

- Stable-Baselines3 PPO: Uses SB3's efficient implementation of PPO with support for parallel environments.

- Parallelized Training: Multiple environments run concurrently for faster experience collection and policy updates.

- Checkpoints: Model checkpoints are saved during training for resuming or evaluating at later stages.

- Experiment Tracking with wandb: Real-time tracking of rewards, losses, and hyperparameters for experiment reproducibility.

---

## Main Libraries Used:

| Library              | Purpose                                                  |
|----------------------|----------------------------------------------------------|
| `stable-baselines3` | RL algorithms like PPO, DQN, A2C (here, PPO with CNN policy) |
| `ale-py`            | Interface for Atari environments like Pong               |
| `wandb`             | Logging training metrics, models, videos, configs        |
| `gymnasium`         | Provides the environment wrappers and simulation interface |
| `moviepy`           | For rendering and saving agent gameplay                  |

---

## Alrorithm - Proximal Policy Optimization (PPO)

PPO is a **policy gradient** method designed to strike a balance between performance and stability during policy updates.

---

### Core Concepts

- **Clipped Objective Function**:  
  Prevents large policy updates by clipping the ratio of new and old policies.

  $$
  L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\ 1 - \epsilon,\ 1 + \epsilon)\hat{A}_t \right) \right]
  $$

  where

  $$
  r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
  $$

  and \( \hat{A}_t \) is the advantage estimate.


- **Advantage Estimation**:  
  Uses **Generalized Advantage Estimation (GAE)** for low-variance, high-bias advantage calculation.

- **Entropy Bonus**:  
  Encourages exploration by penalizing certainty (low entropy) in action distributions.

- **Policy & Value Networks**:  
  PPO uses **actor-critic architecture**, with a shared or separate policy (actor) and value (critic) networks.

---

## Training Setup
- Environment: ALE/Pong-v5 from Gymnasium (Atari Learning Environment).
- Hardware: Google Colab with NVIDIA A100 GPU.
- Total Timesteps: 1 million (1e6)
- Training Framework: Stable-Baselines3 (sb3)
- Experiment Tracking: Integrated with Weights & Biases (wandb) for logging and visualization.

---

## Evaluation
After training, the model is evaluated by:
- Reloading the trained model
- Wrapping the environment for video recording
- Running the policy for 2000 steps
- Saving gameplay as .mp4 for review

See the evaluation video on the performance of our agent for 2000 steps' policy: [https://drive.google.com/file/d/10HKxVAHGWmyz7Y70wlMRAeJtNksowCNu/view?usp=sharing](https://drive.google.com/file/d/10HKxVAHGWmyz7Y70wlMRAeJtNksowCNu/view?usp=sharing)
