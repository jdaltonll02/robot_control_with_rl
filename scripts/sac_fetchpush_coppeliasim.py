# SAC (Soft Actor-Critic) adapted for FetchPushCoppeliaSim-v0
#
# Minimal-diff variant of sac_fetchpush.py, pointed at the CoppeliaSim-backed environment
# instead of MuJoCo. See docs/08-coppeliasim-variant.md — the CoppeliaSim scene must already
# be built and running before this script is run. EXPERIMENTAL / UNVERIFIED: nothing here has
# been run against a live CoppeliaSim instance.
#
# The only differences from sac_fetchpush.py are: the environment import/registration (this
# file), the `env_id` default, and nothing else — the SAC algorithm, HER integration, and
# --monitor-every viewer are all identical, since they're already generic over env_id/env_kwargs.
#
# Based on CleanRL's sac_continuous_action.py
# Original: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
# Docs: https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
#
# Usage:
#   # Smoke test (see docs/08-coppeliasim-variant.md before running):
#   python scripts/sac_fetchpush_coppeliasim.py --reward-type sparse --her --total-timesteps 300 --learning-starts 0
#
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer

# Reused, not copied: pure math on (achieved_goal, desired_goal), zero MuJoCo dependency.
from fetch_push_env import FetchPushFlatWrapper
# Register the CoppeliaSim-backed FetchPush environment (the only import that differs from
# sac_fetchpush.py, which registers the MuJoCo one instead).
from fetch_push_env_coppeliasim import register_fetch_push_coppeliasim_envs
register_fetch_push_coppeliasim_envs()


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rl-fetch-push"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (NOT supported yet for the
    CoppeliaSim backend — see docs/08-coppeliasim-variant.md, "known gaps")"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    monitor_every: int = 0
    """if > 0, every this many timesteps, pause training and show one live rollout with
    the current policy, then resume headless training (0 = disabled). For CoppeliaSim this
    reuses the same running instance rather than opening a second window — verify this in
    the smoke test rather than assuming it."""

    # Environment arguments
    env_id: str = "FetchPushCoppeliaSim-v0"
    """the environment id"""
    reward_type: str = "sparse"
    """reward type for FetchPush (sparse, dense_basic, progress_bonus, energy_efficient —
    multi_component is not ported for this backend, same as the MuJoCo original with --her)"""
    num_envs: int = 1
    """the number of parallel game environments (CoppeliaSim variant is scoped to 1 — true
    vectorization would need multiple CoppeliaSim processes on distinct ZMQ ports)"""

    # HER arguments
    her: bool = False
    """enable Hindsight Experience Replay"""
    gradient_steps: int = 1
    """number of gradient updates per env step (increase for HER, e.g. 4-40)"""

    # Domain randomization arguments
    randomize: bool = False
    """enable domain randomization (randomize physics at each episode reset)"""
    mass_range: tuple = (1.0, 1.0)
    """range [min, max] for object mass multiplier when randomize=True"""
    friction_range: tuple = (1.0, 1.0)
    """range [min, max] for friction multiplier when randomize=True (VERIFY: friction
    randomization is a documented gap for the CoppeliaSim backend, see docs)"""
    size_range: tuple = (1.0, 1.0)
    """range [min, max] for object size multiplier when randomize=True"""

    # Algorithm specific arguments
    total_timesteps: int = 250000
    """total timesteps of the experiments (start much smaller — see docs, throughput risk —
    and size this to your measured steps/sec before committing to a long run)"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.95
    """the discount factor gamma"""
    tau: float = 0.05
    """target smoothing coefficient"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    random_eps: float = 0.3
    """probability of taking a completely random action (epsilon-greedy exploration)"""
    learning_starts: int = 5000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def make_env(env_id, seed, idx, capture_video, run_name, env_kwargs=None):
    def thunk():
        kw = env_kwargs or {}
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **kw)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **kw)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # Pass reward type and domain randomization settings to the environment.
    env_kwargs = {
        "reward_type": args.reward_type,
        "randomize": args.randomize,
        "mass_range": list(args.mass_range),
        "friction_range": list(args.friction_range),
        "size_range": list(args.size_range),
    }
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, env_kwargs=env_kwargs)
         for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    # Optional live viewer for periodic visual monitoring during training (see --monitor-every).
    # For CoppeliaSim this connects to the same running instance rather than a second scene.
    monitor_env = gym.make(args.env_id, **env_kwargs) if args.monitor_every > 0 else None

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32

    if args.her:
        from her_replay_buffer import HERReplayBuffer
        rb = HERReplayBuffer(
            buffer_size=args.buffer_size,
            obs_dim=np.array(envs.single_observation_space.shape).prod(),
            action_dim=np.prod(envs.single_action_space.shape),
            compute_reward_fn=FetchPushFlatWrapper.compute_reward_static,
            reward_type=args.reward_type,
            device=device,
        )
    else:
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            n_envs=args.num_envs,
            handle_timeout_termination=False,
        )

    start_time = time.time()

    if args.her:
        episode_buffer = {
            "obs": [], "action": [], "next_obs": [],
            "reward": [], "done": [],
        }

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # Epsilon-greedy: with probability random_eps, take a random action
            if np.random.random() < args.random_eps:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            fi = infos["final_info"]
            if isinstance(fi, dict) and "episode" in fi:
                # gymnasium >= 1.0 SAME_STEP mode: fi is a vectorized dict
                print(f"global_step={global_step}, episodic_return={fi['episode']['r'][0]:.2f}")
                writer.add_scalar("charts/episodic_return", fi["episode"]["r"][0], global_step)
                writer.add_scalar("charts/episodic_length", fi["episode"]["l"][0], global_step)
                # Log success rate from the final info dict if available
                if "is_success" in fi:
                    writer.add_scalar("charts/success_rate", float(fi["is_success"][0]), global_step)
            elif isinstance(fi, (list, np.ndarray)):
                # gymnasium < 1.0: fi is a list of per-env info dicts
                for info in fi:
                    if info is not None and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        if "is_success" in info:
                            writer.add_scalar("charts/success_rate", float(info["is_success"]), global_step)
                        break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_obs`
        # (gymnasium >= 1.0 uses "final_obs" key with SAME_STEP autoreset mode)
        real_next_obs = next_obs.copy()
        for final_obs_key in ("final_obs", "final_observation"):
            if final_obs_key in infos:
                for idx, trunc in enumerate(truncations):
                    if trunc and infos[final_obs_key][idx] is not None:
                        real_next_obs[idx] = infos[final_obs_key][idx]
                break

        if args.her:
            # HER: accumulate transitions in episode buffer.
            # Squeeze the num_envs axis (always 1 here) so each entry matches
            # HERReplayBuffer's documented per-timestep shapes: (obs_dim,), (action_dim,),
            # and scalar reward/done. Storing the raw vectorized (1, ...) shapes instead
            # breaks HERReplayBuffer's goal-relabeling indexing and reward/done assignment.
            episode_buffer["obs"].append(obs[0].copy())
            episode_buffer["action"].append(actions[0].copy())
            episode_buffer["next_obs"].append(real_next_obs[0].copy())
            episode_buffer["reward"].append(rewards[0].copy())
            episode_buffer["done"].append(terminations[0].copy())

            # Check if episode ended for any env
            for env_idx in range(envs.num_envs):
                if terminations[env_idx] or truncations[env_idx]:
                    # Store the completed episode
                    rb.store_episode({
                        "obs": np.array(episode_buffer["obs"]),
                        "action": np.array(episode_buffer["action"]),
                        "next_obs": np.array(episode_buffer["next_obs"]),
                        "reward": np.array(episode_buffer["reward"]),
                        "done": np.array(episode_buffer["done"]),
                    })
                    # Reset episode buffer for this env
                    episode_buffer = {
                        "obs": [], "action": [], "next_obs": [],
                        "reward": [], "done": [],
                    }
                    break  # Assuming single env for simplicity; adjust for multi-env if needed
        else:
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # Periodic live viewer rollout: pause training briefly to watch the current
        # policy act, then resume. Uses the deterministic (mean) action, same as eval.
        if monitor_env is not None and global_step > 0 and global_step % args.monitor_every == 0:
            print(f"[monitor] step {global_step}: showing live rollout...")
            m_obs, _ = monitor_env.reset()
            m_done = False
            while not m_done:
                with torch.no_grad():
                    _, _, m_action = actor.get_action(torch.Tensor(m_obs).unsqueeze(0).to(device))
                m_obs, _, m_term, m_trunc, _ = monitor_env.step(m_action.squeeze(0).cpu().numpy())
                m_done = m_term or m_trunc

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
          for _grad_step in range(args.gradient_steps):
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(actor.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if monitor_env is not None:
        monitor_env.close()
    writer.close()
