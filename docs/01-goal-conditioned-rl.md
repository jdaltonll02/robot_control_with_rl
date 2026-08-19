# Goal-Conditioned RL & the Sparse Reward Problem

## The task

**FetchPush** (from [Gymnasium-Robotics](https://robotics.farama.org/envs/fetch/)) simulates
a 7-DOF Fetch robot arm that must push a cube across a table to a target position. Every
episode samples a new target, which makes this a **goal-conditioned** task: the optimal
policy is a function of both the current state and the goal, `π(a | s, g)`, not just the
state.

| Property | Value |
|---|---|
| Episode length | 50 timesteps |
| Action space | 4-dim continuous: 3D end-effector velocity + gripper command |
| Default success threshold | object within 5cm of goal |
| Default reward | sparse: `0` if success else `-1` |

## Why goal-conditioned tasks default to sparse reward

The natural reward for "did you achieve the goal?" is binary. A dense, hand-crafted
alternative (e.g. negative distance to goal) is tempting, but it has to be defined
per-goal — and if you're going to relabel goals after the fact (which is exactly what HER
does), your reward function needs to be *recomputable from any goal*, not baked into the
rollout. That constraint is what makes sparse reward the natural starting point, and it's
revisited in [Reward Engineering](03-reward-engineering.md).

## The exploration problem

With a sparse reward and random exploration, the probability of the 7-DOF arm *accidentally*
pushing the cube within 5cm of an arbitrary target in 50 steps is low — empirically, about
**5%** in this codebase (see `results/sac_baseline_results.json`). That means:

- ~95% of episodes return a constant `-1` at every timestep.
- The replay buffer fills with transitions that carry no information about *how* to reach
  a goal — only that this particular attempt failed.
- Standard off-policy algorithms (SAC, DDPG) have no gradient signal to climb. Training
  stalls at the random-policy success rate no matter how long you run it.

This is a structural problem, not a hyperparameter problem — more timesteps, a bigger
network, or a different learning rate don't fix it, because the reward signal itself is
uninformative for the vast majority of experience collected. This is what
[Hindsight Experience Replay](02-her.md) is designed to fix: it doesn't change the reward
function or the algorithm, it changes what the replay buffer teaches the agent by
relabeling *which goal* a transition should be judged against.

## Observation and action layout

The raw `FetchPush-v4` environment returns a `Dict` observation
(`observation`, `desired_goal`, `achieved_goal`). `FetchPushFlatWrapper`
(`scripts/fetch_push_env.py`) flattens this into a single 31-dim vector so it's compatible
with CleanRL's single-file SAC/DDPG implementations, which expect a `Box` observation space:

```
flattened observation (31,)
┌─────────────────────────────┬───────────────┬───────────────┐
│ [0:25] robot observation    │ [25:28]       │ [28:31]       │
│ joint pos/vel, gripper,     │ desired_goal  │ achieved_goal │
│ object pos/vel              │ (x, y, z)     │ (x, y, z)     │
└─────────────────────────────┴───────────────┴───────────────┘
```

`achieved_goal` is simply the object's current position — it's not a separate sensor, it's
a slice of the same state used elsewhere in the observation. That's what makes HER goal
relabeling possible: any timestep's `achieved_goal` is a valid stand-in for `desired_goal`
at an earlier timestep, because they live in the same space.

```
action (4,)
┌───────────────────┬───────────┐
│ [0:3]              │ [3]       │
│ end-effector        │ gripper   │
│ velocity (dx,dy,dz) │ command   │
└───────────────────┴───────────┘
```

The gripper command is part of the action space but isn't functionally required for
*pushing* (as opposed to picking) — the arm pushes the cube using its body/gripper as a
plow, so most of the useful control signal is in the first 3 dimensions.

Next: **[Hindsight Experience Replay →](02-her.md)**
