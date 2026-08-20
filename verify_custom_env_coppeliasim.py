"""
Sanity check + throughput probe for the CoppeliaSim-backed FetchPush environment.

CoppeliaSim must already be running with the scene from docs/08-coppeliasim-variant.md built
and loaded before running this script — it connects to a running instance, it doesn't launch
one. See that doc for the full setup story; nothing here has been run against a live instance.

Run this BEFORE any training: it prints a measured steps/sec, which is the input to the
throughput go/no-go decision in docs/08-coppeliasim-variant.md (250_000 / measured_steps_per_sec
gives the wall-clock floor for a full run).
"""

import time

from scripts.fetch_push_env_coppeliasim import register_fetch_push_coppeliasim_envs
register_fetch_push_coppeliasim_envs()

import gymnasium as gym

N_RESETS = 20
N_STEPS = 200

env = gym.make("FetchPushCoppeliaSim-v0", reward_type="sparse")
obs, info = env.reset()
print(f"Flat observation shape: {obs.shape}")  # expect (31,)
print(f"Action space: {env.action_space}")  # expect Box(-1, 1, (4,))

print(f"\nTiming {N_RESETS} resets...")
start = time.time()
for _ in range(N_RESETS):
    env.reset()
reset_elapsed = time.time() - start
print(f"  {N_RESETS} resets in {reset_elapsed:.2f}s ({reset_elapsed / N_RESETS * 1000:.1f} ms/reset)")

print(f"\nTiming {N_STEPS} steps...")
start = time.time()
for _ in range(N_STEPS):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        env.reset()
step_elapsed = time.time() - start
steps_per_sec = N_STEPS / step_elapsed
print(f"  {N_STEPS} steps in {step_elapsed:.2f}s ({steps_per_sec:.1f} steps/sec)")

print(f"\nAt {steps_per_sec:.1f} steps/sec, a 250,000-step run would take "
      f"{250_000 / steps_per_sec / 3600:.1f} hours wall-clock.")
print("Compare against docs/08-coppeliasim-variant.md's throughput risk section before "
      "committing to a long training run.")

env.close()
