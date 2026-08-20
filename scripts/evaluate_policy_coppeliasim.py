#!/usr/bin/env python3
"""
Evaluation script for trained RL policies on FetchPushCoppeliaSim-v0.

Minimal-diff variant of evaluate_policy.py, pointed at the CoppeliaSim-backed environment
instead of MuJoCo. See docs/08-coppeliasim-variant.md — the CoppeliaSim scene must already be
built and running before this script is run. EXPERIMENTAL / UNVERIFIED: nothing here has been
run against a live CoppeliaSim instance.

SACActorNet, DDPGActorNet, load_cleanrl_model(), and evaluate() are unchanged from
evaluate_policy.py — they're already fully generic over obs/action shapes and don't reference
FetchPushFlatWrapper or MuJoCo directly. Only the registration import and the --env-id default
differ.

Usage:
    python scripts/evaluate_policy_coppeliasim.py \
        --model-path runs/<run-name>/sac-her-coppeliasim.cleanrl_model \
        --env-id FetchPushCoppeliaSim-v0 \
        --reward-type sparse \
        --algorithm SAC \
        --n-episodes 100 \
        --output results/sac_her_coppeliasim_results.json

NOTE: --record-video is NOT supported for this backend yet (CoppeliaSim needs a vision sensor
+ sim.getVisionSensorImg() for rgb_array frames, unlike MuJoCo's direct renderer passthrough —
see docs/08-coppeliasim-variant.md, "known gaps"). Passing it will fail when the env tries to
construct with render_mode="rgb_array".
"""

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Register the CoppeliaSim-backed FetchPush environment (the only import that differs from
# evaluate_policy.py, which registers the MuJoCo one instead).
from fetch_push_env_coppeliasim import register_fetch_push_coppeliasim_envs
register_fetch_push_coppeliasim_envs()


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class _VecEnvShim:
    """Minimal shim so Actor can be instantiated with a plain (non-vectorised) env."""
    def __init__(self, env):
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space


class SACActorNet(nn.Module):
    """Mirrors the Actor class in sac_fetchpush_coppeliasim.py (and sac_fetchpush.py) exactly."""
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        act_dim = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)
        self.register_buffer("action_scale",
            torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",
            torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class DDPGActorNet(nn.Module):
    """Mirrors the Actor class in ddpg_fetchpush_coppeliasim.py (and ddpg_fetchpush.py) exactly."""
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        act_dim = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, act_dim)
        self.register_buffer("action_scale",
            torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",
            torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


def load_cleanrl_model(model_path: str, env: gym.Env, algorithm: str = "SAC"):
    """
    Load a CleanRL saved actor.
    SAC saves actor.state_dict() directly.
    DDPG saves (actor.state_dict(), qf1.state_dict()) as a tuple.
    """
    vec_env = _VecEnvShim(env)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if algorithm.upper() == "DDPG":
        # DDPG checkpoint is a tuple (actor_sd, qf1_sd)
        actor_sd = state[0] if isinstance(state, (tuple, list)) else state
        actor = DDPGActorNet(vec_env)
    else:
        actor_sd = state
        actor = SACActorNet(vec_env)
    # action_scale/action_bias are deterministic functions of the action space bounds,
    # not learned parameters — always let the freshly-constructed model derive them from
    # this env rather than loading them.
    actor_sd = {k: v for k, v in actor_sd.items() if k not in ("action_scale", "action_bias")}
    actor.load_state_dict(actor_sd, strict=False)
    actor.eval()
    return actor


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
    parser = argparse.ArgumentParser(description="Evaluate RL policy on FetchPushCoppeliaSim")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="FetchPushCoppeliaSim-v0")
    parser.add_argument("--reward-type", type=str, default="sparse")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--record-video", action="store_true",
                         help="NOT supported yet for this backend — see module docstring")
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
    model = load_cleanrl_model(args.model_path, dummy_env, algorithm=args.algorithm)
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
        "experiment": f"reward_{args.reward_type}_{args.algorithm.lower()}_coppeliasim",
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
        "notes": "CoppeliaSim backend (experimental)",
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
