#!/usr/bin/env python3
"""
Evaluation script for trained RL policies on FetchPushFlat-v0.

Loads a trained CleanRL model, runs evaluation episodes, computes metrics,
and saves results in the required JSON format.

Usage:
    python scripts/evaluate_policy.py \
        --model-path runs/<run-name>/sac_fetchpush.cleanrl_model \
        --env-id FetchPushFlat-v0 \
        --reward-type sparse \
        --algorithm SAC \
        --n-episodes 100 \
        --output results/sac_her_results.json \
        --record-video \
        --video-dir videos/

NOTE: This is a helper script. Candidates may evaluate differently.
What matters is the output JSON format.
"""

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

# Register custom env
from fetch_push_env import register_fetch_push_envs
register_fetch_push_envs()


def load_cleanrl_model(model_path: str, env: gym.Env):
    """
    Load a CleanRL saved model.

    NOTE TO CANDIDATES: CleanRL's latest scripts save agent.state_dict()
    (not the full model object). You will need to:
    1. Define or import the same Agent/Actor class used during training
    2. Instantiate it with the correct observation/action dimensions
    3. Call model.load_state_dict(torch.load(model_path))

    This function provides a basic fallback that tries torch.load() directly.
    You should replace or extend it to match your training script's Agent class.
    """
    try:
        loaded = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict):
            raise ValueError(
                "Model file contains a state_dict, not a full model. "
                "You need to reconstruct the Agent class and call "
                "agent.load_state_dict(state_dict). See the SAC/DDPG "
                "training script for the Agent/Actor class definition."
            )
        return loaded
    except Exception as e:
        print(f"Could not load model: {e}")
        print("You need to adjust the loading code for your CleanRL version.")
        print("See README for guidance on model loading.")
        raise


def evaluate(
    model,
    env_id: str,
    n_episodes: int = 100,
    record_video: bool = False,
    video_dir: str = "videos/",
    seed: int = 42,
    **env_kwargs,
):
    """Run evaluation episodes and collect metrics."""
    if record_video:
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep: ep < 10,  # Record first 10 episodes
        )
    else:
        env = gym.make(env_id, **env_kwargs)

    successes = []
    episode_returns = []
    episode_lengths = []
    episode_energies = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_length = 0
        ep_energy = 0.0

        while not done:
            # Get action from model
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                if hasattr(model, "get_action"):
                    action, _, _ = model.get_action(obs_tensor)
                    action = action.squeeze(0).numpy()
                elif hasattr(model, "actor"):
                    action = model.actor(obs_tensor).squeeze(0).numpy()
                else:
                    # Fallback: try calling the model directly
                    action = model(obs_tensor).squeeze(0).numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_length += 1
            ep_energy += np.sum(action[:3] ** 2)  # L2 norm of end-effector actions
            done = terminated or truncated

        successes.append(float(info.get("is_success", False)))
        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        episode_energies.append(ep_energy / max(ep_length, 1))

    env.close()

    return {
        "success_rate": float(np.mean(successes)),
        "mean_episode_return": float(np.mean(episode_returns)),
        "std_episode_return": float(np.std(episode_returns)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_energy": float(np.mean(episode_energies)),
        "std_energy": float(np.std(episode_energies)),
        "n_episodes": n_episodes,
        "per_episode_success": successes,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL policy on FetchPushFlat")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="FetchPushFlat-v0")
    parser.add_argument("--reward-type", type=str, default="sparse")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default="videos/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithm", type=str, default="SAC")
    parser.add_argument("--total-timesteps", type=int, default=0)
    parser.add_argument("--training-wall-time", type=float, default=0.0)
    parser.add_argument("--hardware", type=str, default="unknown")
    # Domain randomization eval params
    parser.add_argument("--object-mass-multiplier", type=float, default=1.0)
    parser.add_argument("--friction-multiplier", type=float, default=1.0)
    parser.add_argument("--object-size-multiplier", type=float, default=1.0)
    args = parser.parse_args()

    env_kwargs = {
        "reward_type": args.reward_type,
        "object_mass_multiplier": args.object_mass_multiplier,
        "friction_multiplier": args.friction_multiplier,
        "object_size_multiplier": args.object_size_multiplier,
    }

    print(f"Loading model from: {args.model_path}")
    dummy_env = gym.make(args.env_id, **env_kwargs)
    model = load_cleanrl_model(args.model_path, dummy_env)
    dummy_env.close()

    print(f"Evaluating for {args.n_episodes} episodes...")
    start = time.time()
    metrics = evaluate(
        model,
        args.env_id,
        n_episodes=args.n_episodes,
        record_video=args.record_video,
        video_dir=args.video_dir,
        seed=args.seed,
        **env_kwargs,
    )
    eval_time = time.time() - start

    results = {
        "experiment": f"reward_{args.reward_type}_{args.algorithm.lower()}",
        "algorithm": args.algorithm,
        "reward_type": args.reward_type,
        "env_id": args.env_id,
        "success_rate": metrics["success_rate"],
        "mean_episode_return": metrics["mean_episode_return"],
        "mean_episode_length": metrics["mean_episode_length"],
        "mean_energy": metrics["mean_energy"],
        "total_timesteps": args.total_timesteps,
        "training_wall_time_minutes": args.training_wall_time,
        "n_eval_episodes": args.n_episodes,
        "domain_randomization": {
            "object_mass_multiplier": args.object_mass_multiplier,
            "friction_multiplier": args.friction_multiplier,
            "object_size_multiplier": args.object_size_multiplier,
        },
        "hardware": args.hardware,
        "seed": args.seed,
        "notes": "",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Success rate: {metrics['success_rate']:.1%}")
    print(f"Mean return: {metrics['mean_episode_return']:.2f}")
    print(f"Mean energy: {metrics['mean_energy']:.4f}")
    print(f"Eval time: {eval_time:.1f}s")


if __name__ == "__main__":
    main()
