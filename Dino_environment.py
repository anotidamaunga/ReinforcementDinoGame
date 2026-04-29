"""

BEFORE RUNNING
--------------
  1. Open Chrome -> chrome://dino (or disconnect internet).
  2. Run `python dino_env.py` to open a calibration screenshot.
"""

import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
import mss
import pynput.keyboard as kb


GAME_REGION = {"top": 300, "left": 400, "width": 600, "height": 150}
GAME_OVER_PIXEL_XY = (230, 77)   # (x, y) relative to GAME_REGION
GAME_OVER_PIXEL_THRESHOLD = 200  # brightness above this -> game over


FRAME_W      = 84    # paper: 84x84
FRAME_H      = 84
FRAME_SKIP   = 4     # paper: repeat action for k=4 frames
JUMP_HOLD    = 0.08  # seconds to hold the space bar for a jump

ACTIONS = {0: "nothing", 1: "jump", 2: "duck"}


class DinoEnv(gym.Env):
    """
    Observation : (84, 84, 1) uint8 single grayscale frame.
                  Frame *stacking* is applied externally via VecFrameStack.
    Action      : Discrete(3) - 0 nothing | 1 jump | 2 duck
    Reward      : +1 per FRAME_SKIP frames survived, clipped to [-1, +1].
                  Game-over gives -1 (clipped from any negative penalty).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        # Single frame - stacking is done by VecFrameStack outside
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(FRAME_H, FRAME_W, 1),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(ACTIONS))

        self.render_mode = render_mode
        self.sct = mss.mss()
        self.keyboard = kb.Controller()
        self._steps = 0

    #gym inerface

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._press_space()
        time.sleep(0.5)
        self._release_duck()
        self._steps = 0
        obs = self._get_frame()
        return obs, {}

    def step(self, action):

        #Apply action and repeat it for FRAME_SKIP frames (Atari paper Sec 4).

        total_reward = 0.0
        game_over = False

        for _ in range(FRAME_SKIP):
            self._apply_action(action)
            time.sleep(1 / 30)   # ~30fps base tick

            game_over = self._is_game_over()
            total_reward += -10.0 if game_over else 1.0

            if game_over:
                break

        # Reward clipping - paper Sec 5: clip rewards to [-1, +1]
        clipped_reward = float(np.clip(total_reward, -1.0, 1.0))

        self._steps += 1
        obs = self._get_frame()
        return obs, clipped_reward, game_over, False, {}

    def render(self):
        if self.render_mode == "human":
            frame = self._get_frame()
            display = cv2.resize(
                frame[:, :, 0],
                (FRAME_W * 4, FRAME_H * 4),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imshow("Dino - Agent View (84x84)", display)
            cv2.waitKey(1)

    def close(self):
        self._release_duck()
        cv2.destroyAllWindows()



    def _get_frame(self):
        #screen capture and scaling 4 frames into one like the paper said
        raw = np.array(self.sct.grab(GAME_REGION))
        gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(gray, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]   # (84, 84, 1)

    def _is_game_over(self):
        raw = np.array(self.sct.grab(GAME_REGION))
        x, y = GAME_OVER_PIXEL_XY
        brightness = int(raw[y, x, 1])
        return brightness > GAME_OVER_PIXEL_THRESHOLD


#keyboard control
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
    print("Calibration mode - make sure Chrome/Dino is visible.")
    print(f"Capture region : {GAME_REGION}")
    print(f"Game-over pixel: {GAME_OVER_PIXEL_XY}\n")

    with mss.mss() as sct:
        for i in range(5):
            img = np.array(sct.grab(GAME_REGION))
            x, y = GAME_OVER_PIXEL_XY
            b, g, r, _ = img[y, x]
            print(f"[{i+1}/5] Pixel at {GAME_OVER_PIXEL_XY}: "
                  f"B={b} G={g} R={r}  brightness~{g}")

            if i == 0:
                cv2.imwrite(
                    "calibration_screenshot.png",
                    cv2.cvtColor(img, cv2.COLOR_BGRA2BGR),
                )
                print("      -> Saved calibration_screenshot.png\n")

            time.sleep(1)

    print("Adjust GAME_REGION / GAME_OVER_PIXEL_XY in dino_env.py if needed.")