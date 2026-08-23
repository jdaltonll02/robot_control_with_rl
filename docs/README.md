# Documentation

Detailed, diagrammed walkthrough of the methodology and pipeline in this repository.
Start at the top and work down, or jump straight to the piece you care about.

1. **[Goal-Conditioned RL & the Sparse Reward Problem](01-goal-conditioned-rl.md)** —
   why FetchPush is hard for vanilla RL, the observation/action layout, and the
   exploration problem HER is designed to solve.
2. **[Hindsight Experience Replay](02-her.md)** — the "future" relabeling strategy,
   how it's implemented in `HERReplayBuffer`, and how it plugs into training.
3. **[Reward Engineering](03-reward-engineering.md)** — the five reward functions in
   this repo, their formulas, and the constraint HER imposes on reward design.
4. **[SAC & DDPG](04-algorithms.md)** — network architectures, hyperparameters, and
   how the two algorithms differ in practice on this task.
5. **[Domain Randomization](05-domain-randomization.md)** — randomizing object mass,
   friction, and size for sim-to-real robustness.
6. **[Evaluation Pipeline](06-evaluation.md)** — how trained policies are scored and
   the results JSON schema.
7. **[End-to-End Pipeline](07-pipeline.md)** — how all of the above connects, from
   environment reset to a results file on disk.
8. **[CoppeliaSim Variant](08-coppeliasim-variant.md)** — attempted second backend on the
   real Fetch URDF via CoppeliaSim; reused the HER buffer and reward math unmodified, but
   hit an unresolved physics instability — see [09](09-alternative-simulators.md).
9. **[Alternative Simulators — Investigation Notes](09-alternative-simulators.md)** — honest
   record of three simulator attempts (CoppeliaSim, MuJoCo-with-raw-URDF, PyBullet), none
   completed; what was tried, what broke, and why. The working pipeline above was never
   touched by any of this.
10. **[Architecture](architecture.md)** — the static module-boundary view: what each file
    owns, the shared obs/action contract that lets backends swap, and key design decisions.

## Map of the codebase

| File | Role | Covered in |
|------|------|------------|
| `scripts/fetch_push_env.py` | Env wrapper: flattening, rewards, domain randomization | [01](01-goal-conditioned-rl.md), [03](03-reward-engineering.md), [05](05-domain-randomization.md) |
| `scripts/her_replay_buffer.py` | HER replay buffer | [02](02-her.md) |
| `scripts/sac_fetchpush.py` | SAC training loop | [04](04-algorithms.md), [07](07-pipeline.md) |
| `scripts/ddpg_fetchpush.py` | DDPG training loop | [04](04-algorithms.md), [07](07-pipeline.md) |
| `scripts/evaluate_policy.py` | Checkpoint loading + evaluation + results JSON | [06](06-evaluation.md) |
| `scripts/cleanrl_utils/buffers.py` | CleanRL's standard (non-HER) replay buffer | [07](07-pipeline.md) |
| `scripts/fetch_push_env_coppeliasim.py` + `*_coppeliasim.py` variants | Experimental CoppeliaSim backend | [08](08-coppeliasim-variant.md), [architecture](architecture.md) |
