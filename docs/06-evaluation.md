# Evaluation Pipeline

`scripts/evaluate_policy.py` loads a trained CleanRL checkpoint, runs it deterministically
(no exploration noise, no `random_eps`) for a fixed number of episodes, and writes a
standardized results JSON that every other document in `docs/` and the `results/` directory
assumes as the common format.

## Loading a checkpoint

SAC and DDPG save their actors differently, so loading has to branch on `--algorithm`:

| Algorithm | What `torch.save` wrote | How it's loaded |
|---|---|---|
| SAC | `actor.state_dict()` directly | `SACActorNet` (mirrors `sac_fetchpush.py`'s `Actor`) |
| DDPG | `(actor.state_dict(), qf1.state_dict())` tuple | `DDPGActorNet` (mirrors `ddpg_fetchpush.py`'s `Actor`), using only `state[0]` |

`SACActorNet` and `DDPGActorNet` are line-for-line reimplementations of the corresponding
`Actor` classes in the two training scripts — they have to match exactly, since
`load_state_dict()` matches parameters by name and shape. If you change a network's
architecture in a training script, the mirror class here needs the same change or loading
will fail.

A `_VecEnvShim` wraps a single (non-vectorized) `gym.Env` with the
`single_observation_space` / `single_action_space` attributes the `Actor` constructors
expect — the training scripts build actors against a `SyncVectorEnv`, but evaluation runs one
plain environment at a time, so the shim bridges that interface difference without
duplicating the vector-env machinery just to evaluate.

## The evaluation loop

For each of `n_episodes`, `evaluate()`:

1. Resets with a per-episode seed (`seed + ep`) for reproducibility.
2. Steps the loaded actor deterministically — SAC uses `get_action()`'s `mean` output path is
   *not* used here (it calls `get_action`, which still samples from the Gaussian — see note
   below); DDPG calls the actor directly (already deterministic).
3. Accumulates return, episode length, and **energy** (`sum(action[:3] ** 2)` per step,
   averaged over the episode) — the same energy metric referenced by the
   `energy_efficient` reward in [Reward Engineering](03-reward-engineering.md#energy_efficient).
4. Records `is_success` from the final `info` dict, which the environment wrapper sets
   directly (`FetchPushFlatWrapper.step`).

> **Note:** because `evaluate()` calls `model.get_action(obs_tensor)` for any model exposing
> that method (true for SAC's actor), evaluation episodes still sample from the policy's
> Gaussian rather than using its mean action. For strictly deterministic SAC evaluation, use
> the `mean` return value of `get_action()` instead of the sampled `action`.

If `--record-video` is passed, the first 10 episodes are recorded via
`gym.wrappers.RecordVideo` into `--video-dir` (default `videos/`).

## Results JSON schema

```json
{
  "experiment": "sac_her_sparse",
  "algorithm": "SAC",
  "reward_type": "sparse",
  "env_id": "FetchPushFlat-v0",
  "success_rate": 0.85,
  "mean_episode_return": -8.5,
  "mean_episode_length": 42.3,
  "mean_energy": 0.45,
  "total_timesteps": 250000,
  "training_wall_time_minutes": 120.0,
  "n_eval_episodes": 100,
  "domain_randomization": {
    "object_mass_multiplier": 1.0,
    "friction_multiplier": 1.0,
    "object_size_multiplier": 1.0
  },
  "hardware": "Google Colab CPU / NVIDIA T4",
  "seed": 42,
  "notes": ""
}
```

| Field | Meaning |
|---|---|
| `success_rate` | Fraction of episodes ending with `distance_to_goal < 0.05` |
| `mean_episode_return` | Mean cumulative reward per episode (units depend on `reward_type`) |
| `mean_episode_length` | Mean steps per episode (max 50; shorter implies early termination) |
| `mean_energy` | Mean per-step `‖action[:3]‖²`, averaged per episode then across episodes |
| `total_timesteps` / `training_wall_time_minutes` | Provenance of the checkpoint being evaluated, not measured during evaluation itself — pass these in from the training run |
| `domain_randomization` | The exact fixed multipliers this evaluation run applied (see [Domain Randomization](05-domain-randomization.md)) |
| `seed` | Evaluation seed (episode `i` uses `seed + i`) |

`training_wall_time_minutes` and `hardware` aren't measured by `evaluate_policy.py` — they're
passed in via `--training-wall-time` / `--hardware` flags from whatever you tracked during
the training run itself (e.g. from Wandb's run duration).

## Command

```bash
python scripts/evaluate_policy.py \
  --model-path runs/<run-name>/sac_fetchpush.cleanrl_model \
  --env-id FetchPushFlat-v0 \
  --reward-type sparse \
  --algorithm SAC \
  --n-episodes 100 \
  --output results/sac_her_results.json \
  --record-video --video-dir videos/
```

Next: **[End-to-End Pipeline →](07-pipeline.md)**
