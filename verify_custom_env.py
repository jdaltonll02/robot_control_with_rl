from scripts.fetch_push_env import register_fetch_push_envs
register_fetch_push_envs()

import gymnasium as gym
env = gym.make("FetchPushFlat-v0", reward_type="sparse")
obs, info = env.reset()
print(f"Flat observation shape: {obs.shape}")  # (31,)
print(f"Action space: {env.action_space}")      # Box(-1, 1, (4,))