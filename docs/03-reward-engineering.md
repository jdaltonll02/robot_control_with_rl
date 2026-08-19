# Reward Engineering

Sparse reward + HER is enough to *solve* FetchPush, but the reward function still shapes how
fast training converges and how good the final behavior looks (smoothness, energy use, path
efficiency). This repo defines five reward types in `FetchPushFlatWrapper._compute_reward`
(`scripts/fetch_push_env.py`), selected via `--reward-type`.

## The HER constraint

HER relabels the goal *after* an episode has already happened, and recomputes reward with
`FetchPushFlatWrapper.compute_reward_static(achieved_goal, desired_goal, reward_type)` — a
static method with **no access to the action, the previous timestep, or anything else in the
live rollout.** Any reward term that depends on those (action energy, progress between
timesteps, gripper position) can be used for the *original* transitions in the episode, but
gets dropped or approximated when HER manufactures a virtual transition.

This is the single most important design constraint on reward functions in a HER setting:
**the reward must be reconstructible from `(achieved_goal, desired_goal)` alone**, even if
richer information is available during rollout collection.

## The five reward types

| `reward_type` | Formula (during rollout) | `compute_reward_static` (used for HER relabeling) |
|---|---|---|
| `sparse` | `0` if `distance < 0.05` else `-1` | same — exactly reconstructible |
| `dense_basic` | `-distance` | same — exactly reconstructible |
| `multi_component` | approach + push + progress + time + energy + success (below) | **not implemented** — raises `ValueError` if used with `--her` |
| `progress_bonus` | `sparse_base + 10·(d_{t-1} - d_t) + 5·is_success` | falls back to sparse: `0` if `distance < 0.05` else `-1` |
| `energy_efficient` | `-distance - 0.1·‖a_{0:3}‖² + 1·is_success` | falls back to dense: `-distance` |

`distance` is the L2 norm between the object's `achieved_goal` and `desired_goal`.

### `sparse`

```python
return 0.0 if is_success else -1.0
```

The default from the FetchPush environment. Exactly recomputable from the two goals, so it's
the safest baseline to pair with HER.

### `dense_basic`

```python
return -distance
```

A continuous gradient toward the goal. Also exactly recomputable — `distance` is a pure
function of `(achieved_goal, desired_goal)`.

### `multi_component`

```python
approach_reward  = -gripper_to_object
push_weight      = max(0, 1.0 - gripper_to_object / 0.1)
push_reward      = -distance * (1.0 + 2.0 * push_weight)
progress_reward  = 20.0 * (prev_distance - distance)
time_penalty     = -0.02
energy_penalty   = -0.05 * sum(action[:3] ** 2)
success_bonus    = 50.0 if is_success else 0.0

reward = approach_reward + push_reward + progress_reward
       + time_penalty + energy_penalty + success_bonus
```

A two-phase shaping reward: first pull the gripper to the object (`approach_reward`), then
weight the object→goal distance more heavily once the gripper is close (`push_reward`), with
a progress bonus, a small per-step time penalty, and an energy penalty on top.

**This reward type is not implemented in `compute_reward_static`.** It depends on
`gripper_to_object`, `action`, and `prev_distance` — none of which are available to a static
function that only sees two goal positions. Using `--reward-type multi_component --her`
together will raise a `ValueError` at relabel time. It's included as a rollout-time reward
for **non-HER** runs, or as a reference for how much shaping richer state access affords
compared to what HER-compatible rewards can express.

### `progress_bonus`

```python
sparse_base     = 0.0 if is_success else -1.0
progress_reward = 10.0 * (prev_distance - distance)
success_bonus    = 5.0 if is_success else 0.0

reward = sparse_base + progress_reward + success_bonus
```

Rewards movement in the right direction on top of the sparse base, without fully committing
to a dense distance signal (which can encourage "coasting" near the goal rather than
achieving it outright).

**HER compatibility:** the progress term `(prev_distance - distance)` needs two consecutive
timesteps and can't be recomputed from a single goal pair, so `compute_reward_static` falls
back to plain sparse for this type. This is a deliberate approximation, not a bug: HER's
virtual transitions are constructed so that the achieved goal *is* the (relabeled) desired
goal at the sampled future timestep — i.e., they're already at or very near success — so the
sparse fallback correctly rewards them as successes without needing the progress term.

### `energy_efficient`

```python
dense_distance = -distance
energy_penalty = -0.1 * sum(action[:3] ** 2)
success_bonus   = 1.0 if is_success else 0.0

reward = dense_distance + energy_penalty + success_bonus
```

Adds an action-energy penalty to the dense distance reward — motivated by the fact that real
actuators have torque limits, so a policy that reaches the goal with small, smooth
end-effector velocities is more likely to transfer to hardware than one that reaches it via
large, jerky commands.

**HER compatibility:** the energy term needs the action vector, which isn't available in
`compute_reward_static`. Relabeled transitions fall back to the dense-distance term alone
(`-distance`) — the energy penalty only ever applies to the original (non-relabeled)
transitions in the episode.

## Adding a new reward type

1. Add the type name to `FetchPushFlatWrapper.REWARD_TYPES`.
2. Add an `elif` branch in `_compute_reward` with full access to `obs_dict`, `action`,
   `distance`, `is_success`, `self._prev_distance`, and `self._step_count`.
3. Add a matching branch in `compute_reward_static` that reconstructs (or reasonably
   approximates) the same reward from `(achieved_goal, desired_goal)` alone — this branch is
   what runs whenever HER relabels a transition into this reward type. If no reasonable
   approximation exists, document the limitation the way `multi_component` does above rather
   than silently returning something inconsistent.

Next: **[SAC & DDPG →](04-algorithms.md)**
