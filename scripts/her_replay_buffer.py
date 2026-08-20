"""
Hindsight Experience Replay (HER) Buffer for Goal-Conditioned RL.

Reference: Andrychowicz et al., "Hindsight Experience Replay", NeurIPS 2017
https://arxiv.org/abs/1707.01495

This module provides a replay buffer that implements the HER relabeling
strategy. HER enables learning from failed episodes by retroactively
replacing the desired goal with goals that the agent actually achieved.

The buffer stores complete episodes and, at sample time, creates virtual
transitions where the desired goal is replaced with an achieved goal from
a future timestep in the same episode. This dramatically increases the
number of "successful" transitions the agent sees, which is critical for
sparse-reward goal-conditioned tasks like FetchPush.

Integration:
    The HERReplayBuffer is a drop-in replacement for CleanRL's ReplayBuffer.
    Its `sample()` method returns `ReplayBufferSamples` (same NamedTuple),
    so the training loop does not need to change.

Usage:
    from her_replay_buffer import HERReplayBuffer

    her_buffer = HERReplayBuffer(
        buffer_size=1_000_000,
        obs_dim=31,
        action_dim=4,
        goal_dim=3,
        compute_reward_fn=FetchPushFlatWrapper.compute_reward_static,
        reward_type="sparse",
        n_sampled_goal=4,
        strategy="future",
        device="cpu",
    )

    # During rollout, collect full episodes:
    episode = {"obs": [...], "action": [...], "next_obs": [...],
               "reward": [...], "done": [...]}
    her_buffer.store_episode(episode)

    # During training, sample with HER relabeling:
    batch = her_buffer.sample(batch_size=256)
    # batch.observations, batch.actions, etc. — same as ReplayBufferSamples
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
import torch


class ReplayBufferSamples(NamedTuple):
    """Must match CleanRL's ReplayBufferSamples for compatibility."""
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class HERReplayBuffer:
    """
    Replay buffer with Hindsight Experience Replay (HER) goal relabeling.

    Key concepts:
        - Episodes are stored as complete trajectories (not individual transitions)
        - At sample time, for each sampled transition, with probability
          k/(k+1) we replace the desired goal with an achieved goal from
          a future timestep in the same episode ("future" strategy)
        - The reward is recomputed using the new goal via compute_reward_fn

    Observation layout (FetchPushFlat-v0, 31-dim):
        [0:25]  robot observation
        [25:28] desired_goal
        [28:31] achieved_goal

    Args:
        buffer_size: Maximum number of transitions to store
        obs_dim: Dimension of the flattened observation (default: 31)
        action_dim: Dimension of the action space (default: 4)
        goal_dim: Dimension of the goal space (default: 3)
        compute_reward_fn: Function(achieved_goal, desired_goal, reward_type) -> float
            Used to recompute rewards after goal relabeling.
            See FetchPushFlatWrapper.compute_reward_static
        reward_type: Reward type string passed to compute_reward_fn
        n_sampled_goal: Number of HER virtual goals per real transition (k).
            With k=4, ~80% of sampled transitions are HER-relabeled.
        strategy: Goal sampling strategy. Only "future" is required.
            "future": sample goal from achieved goals at timesteps t+1..T
                      in the same episode
        device: PyTorch device for returned tensors ("cpu" or "cuda")
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int = 31,
        action_dim: int = 4,
        goal_dim: int = 3,
        compute_reward_fn: Callable | None = None,
        reward_type: str = "sparse",
        n_sampled_goal: int = 4,
        strategy: str = "future",
        device: str = "cpu",
    ):
        assert strategy in ("future",), f"Only 'future' strategy is supported, got '{strategy}'"
        assert compute_reward_fn is not None, "compute_reward_fn is required"

        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.compute_reward_fn = compute_reward_fn
        self.reward_type = reward_type
        self.n_sampled_goal = n_sampled_goal  # k in the paper
        self.strategy = strategy
        self.device = device

        # Goal indices within the flattened observation vector
        # FetchPushFlat-v0 layout: [robot_obs(25), desired_goal(3), achieved_goal(3)]
        self.desired_goal_start = obs_dim - 2 * goal_dim  # 25
        self.desired_goal_end = obs_dim - goal_dim          # 28
        self.achieved_goal_start = obs_dim - goal_dim       # 28
        self.achieved_goal_end = obs_dim                    # 31

        # Internal storage: list of episode dicts
        self.episodes = []
        self.total_transitions = 0

    def store_episode(self, episode: dict) -> None:
        """
        Store a complete episode in the buffer.

        Args:
            episode: Dict with keys:
                "obs":      np.ndarray of shape (T, obs_dim) — observations
                "action":   np.ndarray of shape (T, action_dim) — actions taken
                "next_obs": np.ndarray of shape (T, obs_dim) — next observations
                "reward":   np.ndarray of shape (T,) — rewards received
                "done":     np.ndarray of shape (T,) — done flags

                where T is the episode length.

        This method should:
            1. Append the episode to the internal episode storage
            2. Update the total transition count
            3. If the buffer exceeds buffer_size, remove oldest episodes (FIFO)
        """
        # Store a deep copy to avoid accidental modification
        ep = {k: np.copy(v) for k, v in episode.items()}
        ep_len = ep["obs"].shape[0]
        self.episodes.append(ep)
        self.total_transitions += ep_len
        # FIFO: remove oldest episodes if over buffer_size
        while self.total_transitions > self.buffer_size and self.episodes:
            old_ep = self.episodes.pop(0)
            self.total_transitions -= old_ep["obs"].shape[0]

    def _sample_her_goals(
        self,
        episode_idx: int,
        transition_idx: int,
        n_goals: int,
    ) -> np.ndarray:
        """
        Sample n_goals virtual goals using the "future" strategy.

        For the "future" strategy:
            - Sample goal indices uniformly from {transition_idx + 1, ..., T-1}
              where T is the episode length
            - Return the achieved_goal at each sampled timestep

        Args:
            episode_idx: Index of the episode in the buffer
            transition_idx: Index of the transition within the episode
            n_goals: Number of goals to sample

        Returns:
            np.ndarray of shape (n_goals, goal_dim): Sampled goal positions

        Note:
            The achieved_goal at timestep t is the object's position AFTER
            taking the action at timestep t. For HER "future" strategy,
            we want goals from future timesteps, so we sample from the
            *next_obs* achieved_goal (i.e., the achieved goal after the
            transition).
        """
        ep = self.episodes[episode_idx]
        T = ep["obs"].shape[0]
        # If transition_idx is last, sample from [transition_idx, T)
        start = min(transition_idx + 1, T - 1)
        if start >= T:
            start = T - 1
        if start == T - 1:
            sampled_idx = np.full(n_goals, T - 1)
        else:
            sampled_idx = np.random.randint(start, T, size=n_goals)
        # Extract achieved_goal from next_obs at sampled_idx
        ag = ep["next_obs"][sampled_idx, self.achieved_goal_start:self.achieved_goal_end]
        return ag

    def _recompute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> np.ndarray:
        """
        Recompute the reward for relabeled transitions.

        After replacing the desired goal, the reward must be recomputed
        because the original reward was based on the original goal.

        Args:
            achieved_goal: np.ndarray of shape (batch_size, goal_dim)
                The achieved goal (object position) at the next timestep
            desired_goal: np.ndarray of shape (batch_size, goal_dim)
                The new desired goal (from HER relabeling)

        Returns:
            np.ndarray of shape (batch_size,): Recomputed rewards
        """
        # Try to vectorize, else fallback to loop
        try:
            rewards = self.compute_reward_fn(achieved_goal, desired_goal, self.reward_type)
        except Exception:
            rewards = np.array([
                self.compute_reward_fn(achieved_goal[i], desired_goal[i], self.reward_type)
                for i in range(achieved_goal.shape[0])
            ])
        # compute_reward_static returns a plain scalar for a single (1, goal_dim) pair
        # (its norm has no axis argument), so wrap to guarantee an indexable array.
        return np.atleast_1d(rewards)

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Sample a batch of transitions with HER goal relabeling.

        The sampling procedure:
            1. Sample batch_size transitions uniformly from stored episodes
            2. For each transition, with probability k/(k+1):
                a. Sample a virtual goal using _sample_her_goals (1 goal)
                b. Replace the desired_goal in both obs and next_obs
                c. Recompute the reward using _recompute_reward
            3. Return as ReplayBufferSamples (PyTorch tensors on self.device)

        Args:
            batch_size: Number of transitions to sample

        Returns:
            ReplayBufferSamples with fields:
                observations:      (batch_size, obs_dim) tensor
                actions:           (batch_size, action_dim) tensor
                next_observations: (batch_size, obs_dim) tensor
                dones:             (batch_size, 1) tensor
                rewards:           (batch_size, 1) tensor
        """
        assert len(self.episodes) > 0, "Buffer is empty!"
        obs = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
        actions = np.zeros((batch_size, self.action_dim), dtype=np.float32)
        next_obs = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
        rewards = np.zeros((batch_size, 1), dtype=np.float32)
        dones = np.zeros((batch_size, 1), dtype=np.float32)

        k = self.n_sampled_goal
        relabel_prob = k / (k + 1)
        for i in range(batch_size):
            # Sample episode and transition
            ep_idx = np.random.randint(0, len(self.episodes))
            ep = self.episodes[ep_idx]
            ep_len = ep["obs"].shape[0]
            t_idx = np.random.randint(0, ep_len)

            # Copy original transition
            obs_i = np.copy(ep["obs"][t_idx])
            action_i = np.copy(ep["action"][t_idx])
            next_obs_i = np.copy(ep["next_obs"][t_idx])
            done_i = np.copy(ep["done"][t_idx])
            reward_i = np.copy(ep["reward"][t_idx])

            # HER relabeling
            if np.random.rand() < relabel_prob and t_idx < ep_len - 1:
                # Sample 1 virtual goal from future
                virtual_goal = self._sample_her_goals(ep_idx, t_idx, 1)[0]
                # Replace desired_goal in obs and next_obs
                obs_i[self.desired_goal_start:self.desired_goal_end] = virtual_goal
                next_obs_i[self.desired_goal_start:self.desired_goal_end] = virtual_goal
                # Get achieved_goal from next_obs
                achieved_goal = next_obs_i[self.achieved_goal_start:self.achieved_goal_end]
                # Recompute reward
                reward_i = self._recompute_reward(
                    achieved_goal.reshape(1, -1), virtual_goal.reshape(1, -1)
                )[0]

            obs[i] = obs_i
            actions[i] = action_i
            next_obs[i] = next_obs_i
            dones[i, 0] = done_i
            rewards[i, 0] = reward_i

        # Convert to torch tensors
        obs = torch.tensor(obs, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        next_obs = torch.tensor(next_obs, device=self.device)
        dones = torch.tensor(dones, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)

        return ReplayBufferSamples(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
            dones=dones,
            rewards=rewards,
        )

    def __len__(self) -> int:
        """Return total number of transitions stored."""
        return self.total_transitions