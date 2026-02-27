"""
Hindsight Experience Replay (HER) Buffer for Goal-Conditioned RL.

Reference: Andrychowicz et al., "Hindsight Experience Replay", NeurIPS 2017
https://arxiv.org/abs/1707.01495

This module provides a replay buffer that implements the HER relabeling
strategy. HER enables learning from failed episodes by retroactively
replacing the desired goal with goals that the agent actually achieved.

YOUR TASK: Implement the methods marked with TODO below.

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

        # =====================================================================
        # TODO: Initialize storage for episodes and transitions.
        #
        # You need to store complete episodes so that HER can sample future
        # goals from the same episode. Consider using:
        #   - A list of episode dicts, each containing numpy arrays for
        #     obs, action, next_obs, reward, done
        #   - An index tracking total transitions stored
        #
        # Hint: You'll need to handle the buffer_size limit — when full,
        # remove the oldest episodes first (FIFO).
        # =====================================================================
        raise NotImplementedError("TODO: Initialize episode storage")

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
        # =====================================================================
        # TODO: Implement episode storage.
        #
        # Steps:
        #   1. Store the episode (as-is or copy the arrays)
        #   2. Add episode length to total transition count
        #   3. While total transitions > buffer_size:
        #        - Pop the oldest episode
        #        - Subtract its length from the total count
        # =====================================================================
        raise NotImplementedError("TODO: Implement store_episode")

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
        # =====================================================================
        # TODO: Implement "future" strategy goal sampling.
        #
        # Steps:
        #   1. Get the episode from storage
        #   2. Get the episode length T
        #   3. Sample n_goals indices uniformly from [transition_idx + 1, T)
        #      (use np.random.randint)
        #      If transition_idx is the last step, sample from [transition_idx, T)
        #   4. Extract achieved_goal from next_obs at those indices
        #      achieved_goal = next_obs[sampled_idx, achieved_goal_start:achieved_goal_end]
        #   5. Return array of shape (n_goals, goal_dim)
        # =====================================================================
        raise NotImplementedError("TODO: Implement _sample_her_goals")

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
        # =====================================================================
        # TODO: Recompute rewards using self.compute_reward_fn.
        #
        # For each transition in the batch:
        #   reward = self.compute_reward_fn(achieved_goal[i], desired_goal[i],
        #                                   self.reward_type)
        #
        # Hint: You can vectorize this if your compute_reward_fn supports it,
        # or use a simple loop over the batch.
        # =====================================================================
        raise NotImplementedError("TODO: Implement _recompute_reward")

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
        # =====================================================================
        # TODO: Implement the full HER sampling procedure.
        #
        # Steps:
        #   1. Create output arrays: obs, actions, next_obs, rewards, dones
        #      all of shape (batch_size, ...)
        #
        #   2. For each sample i in range(batch_size):
        #      a. Pick a random episode (uniform over stored episodes)
        #      b. Pick a random transition within that episode
        #      c. Copy obs, action, next_obs, done from that transition
        #
        #      d. Decide whether to relabel (with probability k/(k+1)):
        #         - Draw uniform random in [0, 1)
        #         - If < k/(k+1), apply HER relabeling:
        #           * Sample 1 virtual goal using _sample_her_goals
        #           * Replace desired_goal in obs[i] and next_obs[i]
        #             with the virtual goal
        #           * Get achieved_goal from next_obs[i]
        #           * Recompute reward using _recompute_reward
        #         - Otherwise, keep the original transition as-is
        #           * Copy the original reward
        #
        #   3. Convert numpy arrays to PyTorch tensors on self.device
        #   4. Return ReplayBufferSamples(observations, actions,
        #          next_observations, dones, rewards)
        #
        # Important details:
        #   - The desired_goal in obs is at indices [desired_goal_start:desired_goal_end]
        #   - Make sure to COPY arrays before modifying (avoid corrupting stored data)
        #   - dones should have shape (batch_size, 1), same for rewards
        # =====================================================================
        raise NotImplementedError("TODO: Implement sample with HER relabeling")

    def __len__(self) -> int:
        """Return total number of transitions stored."""
        # =====================================================================
        # TODO: Return the total number of transitions across all episodes.
        # =====================================================================
        raise NotImplementedError("TODO: Implement __len__")
