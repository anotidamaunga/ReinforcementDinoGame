"""
nature_cnn.py — DeepMind NatureCNN architecture from:

  "Human-level control through deep reinforcement learning"
  Mnih et al., Nature 2015.  https://doi.org/10.1038/nature14236
The original paper used this with DQN. Here we plug it into PPO as a
custom feature extractor — the policy/value heads are handled by SB3.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class NatureCNN(BaseFeaturesExtractor):


    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # observation_space.shape == (n_stack, H, W) after VecTransposeImage
        # → (H, W, n_stack) which SB3 sees as n_channels=n_stack
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            # Conv 1: 32 filters, 8×8, stride 4
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # Conv 2: 64 filters, 4×4, stride 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # Conv 3: 64 filters, 3×3, stride 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flattened size by doing a dry run
        with torch.no_grad():
            dummy = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(dummy).shape[1]

        # Fully-connected layer → 512-dim features (paper: 512 units)
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalise pixel values to [0, 1] (paper pre-processing step)
        return self.linear(self.cnn(observations.float() / 255.0))