"""
play.py — Watch your trained PPO agent play the Chrome Dino game.

USAGE
-----
    python play.py                          # uses best_model.zip
    python play.py --model models/dino_ppo_final.zip
    python play.py --episodes 20
"""

import argparse
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.monitor import Monitor

from Dino_environment import DinoEnv, open_dino_game

DEFAULT_MODEL = "./models/best_model.zip"


def make_env():
    def _init():
        env = DinoEnv(render_mode="human")   # enables the agent-view window
        env = Monitor(env)
        return env
    return _init


def play(model_path: str, n_episodes: int = 10):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run train.py first."
        )

    print(f" Loading model: {model_path}")
    print(f"   Episodes to run: {n_episodes}")
    print("  Opening Chrome → chrome://dino …")
    open_dino_game(wait=3.0)

    vec_env = DummyVecEnv([make_env()])
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model = PPO.load(model_path, env=vec_env)

    scores = []

    for episode in range(1, n_episodes + 1):
        obs = vec_env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += float(reward[0])
            steps += 1

        scores.append(total_reward)
        print(f"  Episode {episode:3d} | Steps: {steps:5d} | Reward: {total_reward:7.1f}")

    vec_env.close()

    print(f"Results over {n_episodes} episodes:")
    print(f"   Mean reward : {sum(scores)/len(scores):.1f}")
    print(f"   Best reward : {max(scores):.1f}")
    print(f"   Worst reward: {min(scores):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trained PPO agent on Chrome Dino")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to the trained model .zip file")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run (default: 10)")
    args = parser.parse_args()

    play(model_path=args.model, n_episodes=args.episodes)