"""
BEFORE RUNNING
--------------
  1. Run `python Dino_environment.py` to open Chrome and launch the Dino game.
  2. Then run `python train.py` to start training.
"""

import time
import subprocess
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mss
import pynput.keyboard as kb
import pynput.mouse as mouse_lib



GAME_REGION = {"top": 145, "left": 150, "width": 900, "height": 160}

# Absolute screen coords used to click Chrome into focus before key presses.
FOCUS_CLICK_XY = (
    GAME_REGION["left"] + GAME_REGION["width"]  // 2,
    GAME_REGION["top"]  + GAME_REGION["height"] // 2,
)

GAME_OVER_PIXEL_XY        = (450, 25)
GAME_OVER_PIXEL_THRESHOLD = 100

FRAME_W    = 84
FRAME_H    = 84
FRAME_SKIP = 4
JUMP_HOLD  = 0.08

ACTIONS = {0: "nothing", 1: "jump", 2: "duck"}

# Path to Chrome executable — update if your installation differs
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def open_dino_game(wait: float = 2.0):
    """Open Chrome to chrome://dino and wait for it to load."""
    subprocess.Popen([CHROME_PATH, "--new-window", "chrome://dino"])
    time.sleep(wait)


class DinoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(FRAME_H, FRAME_W, 1), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.render_mode  = render_mode
        self.sct          = mss.mss()
        self.keyboard     = kb.Controller()
        self.mouse        = mouse_lib.Controller()
        self._steps       = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._release_duck()
        self._focus_chrome()
        time.sleep(0.1)
        self._press_space()
        time.sleep(0.8)
        self._steps = 0
        return self._get_frame(), {}

    def step(self, action):
        total_reward = 0.0
        game_over    = False

        for _ in range(FRAME_SKIP):
            self._apply_action(action)
            time.sleep(1 / 30)
            game_over     = self._is_game_over()
            total_reward += -10.0 if game_over else 1.0
            if game_over:
                break

        clipped_reward = float(np.clip(total_reward, -1.0, 1.0))
        self._steps   += 1
        return self._get_frame(), clipped_reward, game_over, False, {}

    def render(self):
        if self.render_mode == "human":
            frame   = self._get_frame()
            display = cv2.resize(frame[:, :, 0], (FRAME_W * 4, FRAME_H * 4),
                                 interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Dino – Agent View (84×84)", display)
            cv2.waitKey(1)

    def close(self):
        self._release_duck()
        cv2.destroyAllWindows()

    def _focus_chrome(self):
        x, y = FOCUS_CLICK_XY
        self.mouse.position = (x, y)
        self.mouse.click(mouse_lib.Button.left, 1)

    def _get_frame(self):
        raw     = np.array(self.sct.grab(GAME_REGION))
        gray    = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(gray, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]

    def _is_game_over(self):
        raw        = np.array(self.sct.grab(GAME_REGION))
        x, y       = GAME_OVER_PIXEL_XY
        brightness = int(raw[y, x, 1])
        return brightness < GAME_OVER_PIXEL_THRESHOLD

    def _apply_action(self, action):
        if action == 0:
            self._release_duck()
        elif action == 1:
            self._release_duck()
            self.keyboard.press(kb.Key.space)
            time.sleep(JUMP_HOLD)
            self.keyboard.release(kb.Key.space)
        elif action == 2:
            self.keyboard.press(kb.Key.down)

    def _press_space(self):
        self.keyboard.press(kb.Key.space)
        time.sleep(0.05)
        self.keyboard.release(kb.Key.space)

    def _release_duck(self):
        try:
            self.keyboard.release(kb.Key.down)
        except Exception:
            pass


if __name__ == "__main__":
    print("Opening Chrome → chrome://dino …")
    open_dino_game(wait=2.0)
    print("Chrome is ready. Run train.py to start training.")