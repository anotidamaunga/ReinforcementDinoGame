# Train a PPO agent on the Chrome Dino game using the DeepMind Atari paper
# (Mnih et al. 2015) preprocessing pipeline with NatureCNN as feature extractor.

import argparse
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from Dino_environment import DinoEnv
from nature_cnn import NatureCNN

MODELS_DIR        = "./models"
LOGS_DIR          = "./logs"
BEST_MODEL_PATH   = os.path.join(MODELS_DIR, "best_model")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)

N_STACK = 4   # paper: stack 4 frames


def make_env(render_mode=None):
    def _init():
        env = DinoEnv(render_mode=render_mode)
        env = Monitor(env, LOGS_DIR)
        return env
    return _init


def build_env(render_mode=None):
    vec_env = DummyVecEnv([make_env(render_mode)])
    vec_env = VecTransposeImage(vec_env)               # (84,84,1) → (1,84,84)
    vec_env = VecFrameStack(vec_env, n_stack=N_STACK)  # → (4,84,84)
    return vec_env


def train(total_timesteps: int = 1_000_000, resume: bool = False):
    print(f"\n{'='*56}")
    print("  Chrome Dino — PPO + DeepMind NatureCNN")
    print(f"{'='*56}")
    print(f"  Timesteps  : {total_timesteps:,}")
    print(f"  Frame size : 84x84 grayscale  (Atari paper)")
    print(f"  Frame skip : 4                (Atari paper)")
    print(f"  Frame stack: 4                (Atari paper)")
    print(f"  Reward clip: [-1, +1]         (Atari paper)")
    print(f"  CNN        : NatureCNN        (Atari paper)")
    print(f"  Algorithm  : PPO              (replaces DQN)")
    print(f"{'='*56}\n")

    print("Make sure Chrome is open with chrome://dino visible on screen.")
    print("Starting in 5 seconds...\n")
    time.sleep(5)

    env      = build_env()
    eval_env = build_env()

    policy_kwargs = {
        "features_extractor_class":  NatureCNN,
        "features_extractor_kwargs": {"features_dim": 512},
    }

    ppo_kwargs = dict(
        policy        = "CnnPolicy",
        env           = env,
        policy_kwargs = policy_kwargs,

        # ── Rollout ───────────────────────────────────────────────────────────
        # Larger n_steps gives the agent more experience per update — important
        # when each episode is short (dino dies fast early on).
        n_steps    = 2048,
        batch_size = 64,
        n_epochs   = 4,

        # ── Learning ──────────────────────────────────────────────────────────
        learning_rate  = 2.5e-4,
        clip_range     = 0.1,
        max_grad_norm  = 0.5,
        vf_coef        = 0.5,
        gamma          = 0.99,
        gae_lambda     = 0.95,

        # ── Exploration ───────────────────────────────────────────────────────
        # Higher entropy coefficient stops the agent collapsing to "do nothing"
        # early in training. 0.05 forces it to keep trying jumps/ducks until
        # it discovers they avoid death. Reduce toward 0.01 once it's learning.
        ent_coef = 0.05,

        tensorboard_log = LOGS_DIR,
        verbose         = 1,
    )

    if resume and os.path.exists(BEST_MODEL_PATH + ".zip"):
        print(f"Resuming from {BEST_MODEL_PATH}.zip ...")
        model = PPO.load(BEST_MODEL_PATH, env=env)
    else:
        model = PPO(**ppo_kwargs)

    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = MODELS_DIR,
        name_prefix = "dino_ppo",
        verbose     = 1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = MODELS_DIR,
        log_path             = LOGS_DIR,
        eval_freq            = 25_000,
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 1,
    )

    model.learn(
        total_timesteps     = total_timesteps,
        callback            = [checkpoint_cb, eval_cb],
        reset_num_timesteps = not resume,
    )

    final_path = os.path.join(MODELS_DIR, "dino_ppo_final")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO + NatureCNN on Chrome Dino")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--resume",    action="store_true",
                        help="Resume from best_model.zip")
    args = parser.parse_args()
    train(total_timesteps=args.timesteps, resume=args.resume)