import gymnasium as gym
import numpy as np
from gymnasium import spaces

class PointMass2D(gym.Env):
    def __init__(self, goal=(0.0, 0.0), dt=0.1, max_steps=200):
        super().__init__()

        self.goal = np.array(goal, dtype=np.float32)
        self.dt = dt
        self.max_steps = max_steps

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(-4, 4, size=2).astype(np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        self.state = self.state + self.dt * action
        self.steps += 1

        # potential-based reward
        dist = np.linalg.norm(self.state - self.goal)
        reward = -dist**2

        done = dist < 0.1
        truncated = self.steps >= self.max_steps

        return self.state, reward, done, truncated, {}
