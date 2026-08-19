# RL for Robot Control — FetchPush with Hindsight Experience Replay

A from-scratch implementation of **Hindsight Experience Replay (HER)** applied to a simulated
7-DOF Fetch robot arm learning to push a cube to a target location. The task is a classic
example of **goal-conditioned RL with a sparse reward** — one that's effectively unsolvable
with vanilla off-policy RL, and a good testbed for understanding why HER works, how reward
shaping interacts with goal relabeling, and how SAC and DDPG differ in practice.

<table>
<tr>
<td align="center"><b>Without HER (random baseline)</b></td>
<td align="center"><b>With SAC + HER (trained policy)</b></td>
</tr>
<tr>
<td align="center"><img src="assets/fetch_push_random.gif" width="300" alt="Random baseline — robot fails to push object to target"/></td>
<td align="center"><img src="assets/fetch_push_success.gif" width="300" alt="Trained SAC+HER policy — robot successfully pushes object to target"/></td>
</tr>
<tr>
<td align="center">~5% success rate</td>
<td align="center">~99% success rate</td>
</tr>
</table>

## What's in here

- A custom Gymnasium wrapper (`scripts/fetch_push_env.py`) around `FetchPush-v4` that
  flattens the observation, exposes several reward functions, and supports domain
  randomization of object mass/friction/size.
- A **from-scratch HER replay buffer** (`scripts/her_replay_buffer.py`) implementing the
  "future" goal-relabeling strategy from Andrychowicz et al., 2017.
- Single-file **SAC** and **DDPG** training scripts (`scripts/sac_fetchpush.py`,
  `scripts/ddpg_fetchpush.py`), adapted from [CleanRL](https://docs.cleanrl.dev/), with HER
  and domain-randomization integration.
- An evaluation pipeline (`scripts/evaluate_policy.py`) that loads a trained checkpoint,
  runs evaluation episodes, and emits a standardized results JSON.

**For a detailed, diagrammed walkthrough of the methodology and the end-to-end pipeline,
see [`docs/`](docs/README.md).** The docs cover the sparse-reward problem, exactly how HER
relabeling works in this codebase, the SAC/DDPG architectures and hyperparameters, the
reward functions and their HER-compatibility constraints, domain randomization, and the
evaluation pipeline — with diagrams for each.

## Why HER

**FetchPush** is goal-conditioned: every episode has a different target position, and the
natural reward is sparse (0 if the object is within 5cm of the goal, -1 otherwise). With
random exploration, a 7-DOF arm reaches the goal by chance only ~5% of the time, so a
standard replay buffer almost never contains a positive learning signal.

HER's insight: **a failed attempt is a successful attempt at a different goal.** After an
episode fails to reach its intended goal, HER retroactively relabels transitions with a goal
the agent *did* achieve (a future object position from the same episode) and recomputes the
reward accordingly. This turns most failed episodes into useful training data, without
changing the environment or the algorithm's core update rule — it only changes what the
replay buffer returns.

## Repository layout

```
.
├── README.md                    # this file
├── docs/                        # detailed methodology + pipeline documentation
├── scripts/
│   ├── fetch_push_env.py        # env wrapper: flattening, reward functions, domain randomization
│   ├── her_replay_buffer.py     # HER replay buffer (future-strategy relabeling)
│   ├── sac_fetchpush.py         # SAC training script (CleanRL-based)
│   ├── ddpg_fetchpush.py        # DDPG training script (CleanRL-based)
│   ├── evaluate_policy.py       # loads a checkpoint, evaluates, writes results JSON
│   └── cleanrl_utils/buffers.py # CleanRL's standard (non-HER) replay buffer
├── results/                     # evaluation results (JSON, one per experiment)
├── figures/                     # plots (training curves, comparisons)
├── videos/                      # recorded rollouts
├── assets/                      # README gifs
└── verify_custom_env.py         # quick sanity check for the custom env
```

## Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)

# Core dependencies
pip install gymnasium[mujoco] gymnasium-robotics
pip install torch            # CPU is sufficient
pip install tyro wandb tensorboard

# CleanRL's replay buffer utility
mkdir -p cleanrl_utils && touch cleanrl_utils/__init__.py
curl -sL "https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl_utils/buffers.py" \
  -o cleanrl_utils/buffers.py

wandb login
```

Verify the custom environment:

```bash
python verify_custom_env.py
```

Expected output: a flattened `(31,)` observation and a `Box(-1, 1, (4,))` action space.

## Training

```bash
# Baseline: SAC without HER — expect ~5% success (demonstrates the sparse-reward problem)
python scripts/sac_fetchpush.py \
  --reward-type sparse \
  --total-timesteps 250000 \
  --track --wandb-project-name rl-fetch-push \
  --exp-name sac-baseline-no-her --seed 42

# SAC + HER — expect success rate to climb from ~5% to >50% by ~80K steps, >90% by 200K
python scripts/sac_fetchpush.py \
  --reward-type sparse --her \
  --total-timesteps 250000 \
  --track --wandb-project-name rl-fetch-push \
  --exp-name sac-her-sparse --seed 42

# DDPG + HER
python scripts/ddpg_fetchpush.py \
  --reward-type sparse --her \
  --total-timesteps 250000 \
  --track --wandb-project-name rl-fetch-push \
  --exp-name ddpg-her-sparse --seed 42
```

Custom reward types (`dense_basic`, `multi_component`, `progress_bonus`, `energy_efficient`)
are selected with `--reward-type`; see [`docs/03-reward-engineering.md`](docs/03-reward-engineering.md)
for their formulations and HER-compatibility notes. Domain randomization is enabled with
`--randomize --mass-range MIN MAX --friction-range MIN MAX --size-range MIN MAX`; see
[`docs/05-domain-randomization.md`](docs/05-domain-randomization.md).

## Evaluation

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

This produces a JSON with success rate, mean return, episode length, and mean action energy —
see [`docs/06-evaluation.md`](docs/06-evaluation.md) for the full schema and metric definitions.

## Background reading

| Topic | Resource |
|-------|----------|
| HER | [Andrychowicz et al., "Hindsight Experience Replay", NeurIPS 2017](https://arxiv.org/abs/1707.01495) |
| SAC | [Haarnoja et al., "Soft Actor-Critic", 2018](https://arxiv.org/abs/1801.01290) |
| DDPG | [Lillicrap et al., "Continuous control with deep RL", 2015](https://arxiv.org/abs/1509.02971) |
| Multi-goal RL | [Plappert et al., 2018](https://arxiv.org/abs/1802.09464) |
| Domain randomization | [Tobin et al., 2017](https://arxiv.org/abs/1703.06907) |
| CleanRL | [docs.cleanrl.dev](https://docs.cleanrl.dev/) |
| FetchPush environment | [Gymnasium-Robotics Fetch docs](https://robotics.farama.org/envs/fetch/) |
