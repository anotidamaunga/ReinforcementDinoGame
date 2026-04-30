# Reinforcement Dino Game

A reinforcement learning agent that learns to play the Chrome Dinosaur game from raw pixels using PPO and the DeepMind NatureCNN architecture.

---

## How It Works

The agent watches a cropped region of the screen (84×84 grayscale), stacks 4 consecutive frames to perceive motion, and outputs one of three actions — do nothing, jump, or duck. It receives +1 reward for every frame it survives and -10 when it dies. Training uses PPO, the same CNN architecture from DeepMind's 2015 Atari paper.

```
Screen capture (84×84 grayscale)
        ↓
  Stack 4 frames  →  (4, 84, 84)
        ↓
   NatureCNN
   3× Conv layers → 512-dim features
        ↓
  PPO Actor/Critic
  Action: nothing / jump / duck
```

---

## Project Structure

```
ReinforcementDinoGame/
├── Dino_environment.py   — Gym environment (screen capture, keyboard, game-over detection)
├── nature_cnn.py         — DeepMind NatureCNN feature extractor
├── train.py              — PPO training loop with checkpointing
├── play.py               — Run a trained model and watch it play
├── models/               — Saved model checkpoints
└── logs/                 — TensorBoard training logs
```

---

## Setup

**Requirements:** Python 3.10+, Google Chrome

```bash
pip install stable-baselines3[extra] gymnasium opencv-python mss pynput torch tensorboard
```

If `opencv-python` times out, use the headless build instead:
```bash
pip install opencv-python-headless
```

---

## How to view the project

### 1. Launch the game

Opens Chrome directly to `chrome://dino`:

```bash
python Dino_environment.py
```

### 2. Train the agent

```bash
python train.py                        # 1M timesteps (default)
python train.py --timesteps 3000000    # recommended: 3M steps
python train.py --resume               # continue from best_model.zip
```

Checkpoints are saved to `models/` every 50k steps. The best model by eval reward is saved automatically as `models/best_model.zip`.

### 3. Watch it play

```bash
python play.py                              # uses models/best_model.zip
python play.py --model models/dino_ppo_final.zip
python play.py --episodes 20
```

Opens Chrome automatically, loads the model, and runs the specified number of episodes. Prints per-episode stats and a summary at the end.

Monitor training in TensorBoard:
```bash
tensorboard --logdir logs/
```

---

## Architecture

### NatureCNN (`nature_cnn.py`)

Replicates the CNN from [Mnih et al., 2015](https://arxiv.org/abs/1312.5602):

| Layer  | Type    | Filters | Kernel | Stride | Output         |
|--------|---------|---------|--------|--------|----------------|
| Conv 1 | Conv2d  | 32      | 8×8    | 4      | 32 × 20 × 20  |
| Conv 2 | Conv2d  | 64      | 4×4    | 2      | 64 × 9 × 9    |
| Conv 3 | Conv2d  | 64      | 3×3    | 1      | 64 × 7 × 7    |
| FC     | Linear  | —       | —      | —      | 512            |

Input pixels are normalised to [0, 1] before the first convolution.

### PPO Hyperparameters (`train.py`)

| Parameter       | Value  | Notes                                      |
|-----------------|--------|--------------------------------------------|
| `n_steps`       | 2048   | Large buffer for short early episodes      |
| `batch_size`    | 64     |                                            |
| `n_epochs`      | 4      | Gradient updates per rollout               |
| `learning_rate` | 5e-4   |                                            |
| `clip_range`    | 0.1    | Conservative policy updates                |
| `gamma`         | 0.99   |                                            |
| `gae_lambda`    | 0.95   |                                            |
| `ent_coef`      | 0.05   | High entropy encourages early exploration  |
| `vf_coef`       | 0.5    |                                            |
| `max_grad_norm` | 0.5    |                                            |

---

## Environment Details (`Dino_environment.py`)

| Setting              | Value                                    |
|----------------------|------------------------------------------|
| Observation space    | Box(0, 255, shape=(84, 84, 1), uint8)   |
| Action space         | Discrete(3) — nothing / jump / duck     |
| Frame skip           | 4                                        |
| Frame stack          | 4                                        |
| Reward per frame     | +1 (surviving), -10 (game over)         |
| Reward clipping      | [-1, +1]                                 |
| Game region          | top=145, left=150, width=900, height=160 |
| Jump hold duration   | 0.08s                                    |

Game-over is detected by checking the brightness of a fixed pixel where the "GAME OVER" text appears. Adjust `GAME_OVER_PIXEL_XY` or `GAME_OVER_PIXEL_THRESHOLD` in `Dino_environment.py` if detection is unreliable on your screen.

If Chrome is installed in a non-default location, update `CHROME_PATH` in `Dino_environment.py`.

---

## Training Progression

| Timesteps   | Expected Behaviour                       |
|-------------|------------------------------------------|
| 0 – 100k    | Random actions, dies immediately         |
| 100k – 300k | Starts jumping occasionally              |
| 300k – 600k | Clears single cacti reliably             |
| 600k – 1M+  | Handles cactus clusters and pterodactyls |

---

## What I Learned

**Calibration was the wrong starting point.** My initial approach to setting up the environment was centred around calibration — manually finding the exact pixel coordinates where the dino runs and where the "GAME OVER" text appears. This meant every time the window moved or the screen resolution changed, the setup would break. The better solution was to have the script open Chrome and navigate to `chrome://dino` automatically, removing the manual step entirely and making the environment self-contained from the start.

**Bridging a real game and a Gym environment is the hard part.** The RL algorithm itself (PPO) is largely handled by Stable Baselines3. The real challenge was wrapping a live browser game as a proper Gym environment — handling screen capture timing, keyboard input, focus management, and reliable game-over detection. Getting these details right matters as much as the model architecture.

---

## Design Decisions

**Learning rate: 5e-4 instead of the paper's 2.5e-4**

The original Mnih et al. paper uses a learning rate of 2.5e-4, which proved too slow for this environment. Training progress was minimal over the first few hundred thousand steps, so the learning rate was bumped up to 5e-4 to speed up convergence. The tradeoff is that a higher learning rate can make training less stable — policy updates are larger, which risks overshooting good solutions. For a game as simple as Dino compared to full Atari, the faster learning rate is a reasonable trade.

**PPO over DQN**

The paper uses DQN, but PPO was chosen here because it is more stable to tune and less sensitive to hyperparameters. DQN requires a large replay buffer and careful management of the target network update frequency. PPO converges more reliably with less setup, which is a better fit for a project focused on getting a working agent rather than replicating the paper exactly.

---

## References

- Mnih et al. (2015) — *Human-level control through deep reinforcement learning*. https://arxiv.org/abs/1312.5602
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
