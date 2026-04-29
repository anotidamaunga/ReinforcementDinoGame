# ReinforcementDinoGame


An AI agent that learns to play the Chrome Dinosaur game using deep reinforcement learning. The agent is trained entirely from raw pixels — no hardcoded rules, no game state access. It discovers how to jump and duck purely from the reward signal of surviving longer.

---

## How it works

The agent uses **PPO (Proximal Policy Optimization)** with a **NatureCNN** feature extractor, replicating the preprocessing pipeline from DeepMind's 2015 Atari paper *(Mnih et al., "Human-level control through deep reinforcement learning")*.

At each step the agent:
1. Captures a grayscale 84×84 screenshot of the game
2. Stacks 4 consecutive frames to perceive motion and obstacle speed
3. Feeds the stack through the CNN to extract 512 features
4. Chooses an action — nothing, jump, or duck
5. Receives +1 for surviving or -1 for dying, clipped to [-1, +1]

The policy updates every 2048 steps using the collected experience, gradually learning to associate incoming obstacles with the jump action.

---

## Project structure

```
ReinforcementDinoGame/
├── Dino_environment.py   # Gymnasium env — screen capture, keyboard control, game-over detection
├── nature_cnn.py         # DeepMind NatureCNN architecture (3 conv layers + FC)
├── train.py              # PPO training loop with checkpointing and evaluation
├── play.py               # Watch a trained model play
├── models/               # Saved checkpoints and best model
└── logs/                 # TensorBoard training logs
```

---

## Requirements

- Python 3.10+
- Google Chrome (open to `chrome://dino` or disconnect internet)
- Windows (uses `mss` for screen capture and `pynput` for keyboard/mouse control)

Install dependencies:

```bash
pip install stable-baselines3[extra] gymnasium opencv-python mss pynput torch tensorboard tqdm rich
```

---

## Setup

**1. Calibrate the screen region**

Open Chrome with the Dino game visible, then run:

```bash
python Dino_environment.py
```

This saves `calibration_screenshot.png`. Open it and verify the dino and ground line are fully visible. If not, adjust `GAME_REGION` in `Dino_environment.py`:

```python
GAME_REGION = {"top": 145, "left": 150, "width": 900, "height": 160}
```

**2. Verify game-over detection**

Crash the dino deliberately and watch the printed brightness values. The pixel should drop below `100` when "GAME OVER" appears. If not, adjust `GAME_OVER_PIXEL_XY` to point at the centre of the "GAME OVER" text.

---

## Training

Make sure Chrome is visible on screen, then:

```bash
python train.py
```

Optional arguments:

```bash
python train.py --timesteps 3000000   # train for 3M steps (recommended)
python train.py --resume              # continue from best_model.zip
```

Checkpoints are saved to `models/` every 50,000 steps. The best-performing model is saved automatically to `models/best_model.zip`.

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir logs/
```

### What to expect

| Timesteps | Behaviour |
|-----------|-----------|
| 0 – 100k  | Runs straight into cacti — random policy |
| 100k – 300k | Occasional jumps, starts surviving longer |
| 300k – 600k | Consistently clears single cacti |
| 600k – 1M+ | Handles clusters and begins reacting to birds |

Results vary depending on your screen resolution and capture region calibration.

---

## Watching the agent play

```bash
python play.py                                  # uses best_model.zip by default
python play.py --model models/dino_ppo_final.zip
python play.py --episodes 20
```

An 84×84 agent-view window will open alongside Chrome so you can see exactly what the agent sees.

---

## Architecture

```
Input: 4 × 84 × 84 (stacked grayscale frames)
       ↓
Conv2d(4 → 32, kernel=8, stride=4)  + ReLU
       ↓
Conv2d(32 → 64, kernel=4, stride=2) + ReLU
       ↓
Conv2d(64 → 64, kernel=3, stride=1) + ReLU
       ↓
Flatten → Linear(→ 512) + ReLU
       ↓
PPO Actor head  →  Discrete(3) action
PPO Critic head →  Value estimate
```

---

## Key hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `n_steps` | 2048 | More episodes per update — important when early episodes are very short |
| `ent_coef` | 0.05 | Higher entropy encourages the agent to keep exploring jumps early in training |
| `clip_range` | 0.1 | Conservative updates — prevents the policy changing too fast |
| `gamma` | 0.99 | Values future survival highly |
| `frame_skip` | 4 | Matches Atari paper — same action repeated for 4 frames |
| `frame_stack` | 4 | Gives the agent motion information across frames |

---


