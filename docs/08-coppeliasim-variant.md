# CoppeliaSim Variant (experimental, unverified)

A second, independent backend for the same goal-conditioned FetchPush task, running on
CoppeliaSim (EDU) instead of MuJoCo, using the real Fetch robot's URDF. Nothing in the
original MuJoCo pipeline (`scripts/fetch_push_env.py`, `her_replay_buffer.py`,
`sac_fetchpush.py`, `ddpg_fetchpush.py`, `evaluate_policy.py`, `verify_custom_env.py`) is
modified — every file listed below is new.

**Status: none of this has been run against a live CoppeliaSim instance.** The scene itself
(URDF import, IK group, cube/goal objects) has to be built by hand in the CoppeliaSim GUI —
that's not something achievable through file edits — so the Python side below is written
against the *expected* scene layout, not verified against an actual one. Treat this as a
first implementation to smoke-test and adjust, not a finished, working integration. Places
where the exact API depends on your installed CoppeliaSim version/physics engine and
couldn't be confirmed without a running instance are marked `# VERIFY:` in the code.

## Why a second backend

MuJoCo and CoppeliaSim model contact/friction/actuation differently, so a MuJoCo-trained
policy doesn't drop into CoppeliaSim and work — this is training from scratch in a second
simulator, reusing everything that's backend-agnostic (the reward math, the HER buffer, the
SAC/DDPG algorithms) and reimplementing only what's genuinely simulator-specific
(observation construction, action application, domain randomization).

## What's reused vs. reimplemented

| Component | Status |
|---|---|
| `HERReplayBuffer` (`scripts/her_replay_buffer.py`) | **Reused unmodified** — it's generic over `obs_dim`/`action_dim`/`goal_dim`/`compute_reward_fn`, zero MuJoCo coupling |
| `FetchPushFlatWrapper.compute_reward_static` / `get_goal_from_obs` | **Reused by import** — pure math on `(achieved_goal, desired_goal)`, zero sim dependency |
| SAC/DDPG networks, training loop, `--monitor-every` viewer, HER episode collection | **Reused via minimal-diff copies** — already generic over `env_id`/`env_kwargs` |
| Observation construction, action→IK application, domain randomization | **Reimplemented** — genuinely CoppeliaSim-specific |

## Prerequisites

```bash
pip install -r scripts/coppeliasim/requirements_coppeliasim.txt
```

This installs `coppeliasim-zmqremoteapi-client` (the current officially maintained CoppeliaSim
Python API — chosen over PyRep, which is tied to CoppeliaSim ~4.1 and less actively
maintained), plus `pyzmq` and `cbor2`.

## Building the scene (manual, in the CoppeliaSim GUI — this is on you)

1. **Get the Fetch URDF.** Clone the public `fetch_ros` repo somewhere local:
   ```bash
   cd scripts/coppeliasim/assets
   git clone --depth 1 https://github.com/fetchrobotics/fetch_ros.git
   ```
   (no git available? download the ZIP from that URL in a browser instead). This directory
   is *not* committed — treat it like `venv/`, present locally and untracked. It already
   contains a plain, pre-resolved URDF at `fetch_ros/fetch_description/robots/fetch.urdf` —
   **no xacro conversion needed**, that file is ready to use as-is.
2. **Fix up mesh paths** so CoppeliaSim's URDF importer can resolve them (it doesn't
   understand ROS's `package://` URIs — those only mean something inside a full ROS install,
   which this project doesn't have):
   ```bash
   python scripts/coppeliasim/convert_urdf_paths.py \
     --input scripts/coppeliasim/assets/fetch_ros/fetch_description/robots/fetch.urdf \
     --output scripts/coppeliasim/assets/fetch_fixed.urdf \
     --package-name fetch_description \
     --package-dir scripts/coppeliasim/assets/fetch_ros/fetch_description
   ```
4. **Import into CoppeliaSim:** Plugins → URDF Import → select `fetch_fixed.urdf`. This
   produces a tree of link/joint objects for the real 7-DOF Fetch arm + torso + gripper.
5. **Add task objects:**
   - A dynamic, respondable cube — name it `PushObject`.
   - A non-dynamic, non-respondable (purely visual) sphere or cube — name it `GoalMarker`.
6. **Set up the IK group** (the CoppeliaSim analog of MuJoCo's mocap-weld mechanism):
   - Add a dummy welded to the gripper's attachment frame — name it `IKTip`.
   - Add a second, freely-movable dummy that the Python side will reposition every step —
     name it `IKTarget`.
   - Using the IK plugin dialog, create an IK group (suggested name `FetchIK`) whose tip is
     `IKTip` and whose target is `IKTarget`, covering the arm's joint chain.
   - **Fidelity note**: MuJoCo's mocap+weld is resolved *compliantly* by the physics solver
     every one of its 20 internal substeps per control step — it's not a one-shot IK solve.
     `simIK.handleIkGroup` is a one-shot solve per call. The wrapper compensates by calling it
     once per control step to get target joint angles, applying them via
     `sim.setJointTargetPosition` on **position-controlled dynamic joints**, and stepping
     physics several times to let the engine's PD servo converge — not by teleporting joints
     directly. This is the single biggest behavioral difference from the MuJoCo original and
     the most likely thing to need hand-tuning (PD gains, step count) once you can actually
     run it.
7. **Save the scene** as `scripts/coppeliasim/assets/fetch_push_scene.ttt` (also untracked,
   like the URDF assets — binary, local-only).
8. **Update `OBJECT_PATHS`** at the top of `scripts/fetch_push_env_coppeliasim.py` to match
   whatever names/paths your imported model tree actually uses (URDF import doesn't
   necessarily preserve names exactly as listed above — check the scene hierarchy after
   import and adjust the dict, it's the only place these names live).

## Running it

CoppeliaSim must already be running with the scene loaded before you run any Python here —
the wrapper connects to a running instance, it doesn't launch one (see
`scripts/coppeliasim/launch_headless.py` for an optional headless-launch helper once you've
validated the scene interactively).

```bash
# 1. Sanity check + throughput probe
python verify_custom_env_coppeliasim.py

# 2. Smoke test — short run, watch for IK divergence / dropped connections / crashes
python scripts/sac_fetchpush_coppeliasim.py \
  --reward-type sparse --her \
  --total-timesteps 300 --learning-starts 0 \
  --exp-name coppelia-smoke --seed 42

# 3. Full training — only after the throughput gate below says it's practical
python scripts/sac_fetchpush_coppeliasim.py \
  --reward-type sparse --her \
  --total-timesteps <sized to measured steps/sec> \
  --exp-name sac-her-coppeliasim --seed 42

# 4. Evaluate exactly like the MuJoCo checkpoints
python scripts/evaluate_policy_coppeliasim.py \
  --model-path runs/<run-folder>/sac-her-coppeliasim.cleanrl_model \
  --algorithm SAC --n-episodes 100 --output results/sac_her_coppeliasim_results.json
```

## The throughput risk (unresolved — this is the real open question)

The MuJoCo pipeline already only reaches ~20-40 steps/sec in-process on this machine's CPU.
CoppeliaSim's remote API adds real network/IPC round-trip latency on top of that, and it gets
worse if a step issues many separate calls (move IK target, step, read grip pos, read object
pos, read velocities, read joints, ...).

`verify_custom_env_coppeliasim.py` prints a measured steps/sec specifically so you have a real
number before committing to a long run — compute `250_000 / measured_steps_per_sec` and decide
whether that's practical (minutes/hours) or not (many hours/days).

If it's not practical, the primary lever is **reducing round trips, not physics substeps**:
write a small CoppeliaSim-side Lua child script exposing one custom function (e.g.
`getObsAndStep`) called via `sim.callScriptFunction`, so one env `step()` is exactly one ZMQ
round trip instead of six or seven. That Lua script isn't written here — it's a follow-up if
the measured throughput demands it.

## Known gaps

- **Domain randomization friction**: `_apply_domain_randomization_sim` sets mass and scale via
  confirmed real API calls (`sim.setShapeMass`, `sim.scaleObject`), but the friction call is
  marked `# VERIFY:` — the exact parameter/enum depends on which physics engine your scene
  uses (Bullet/ODE/Newton/Vortex), which can only be confirmed by checking your actual
  CoppeliaSim installation.
- **Video recording**: `--capture-video`/`--record-video` aren't supported yet — CoppeliaSim
  needs a vision sensor + `sim.getVisionSensorImg()` for `rgb_array` frames, unlike MuJoCo's
  direct renderer passthrough. Left as a stretch goal.
- **Vectorization**: scoped to `num_envs=1` deliberately — true vectorization would need
  multiple CoppeliaSim processes on distinct ZMQ ports, a separate, larger effort.
- **Coordinate frame**: the imported Fetch model's actual home/gripper pose depends on where
  URDF import places it in the scene, which is *not* MuJoCo's hardcoded offset. The wrapper
  captures this empirically at first reset rather than assuming a fixed value — verify this
  behaves sensibly once you can actually run it.

## File map

| File | Role |
|---|---|
| `scripts/fetch_push_env_coppeliasim.py` | The new `gym.Env`, CoppeliaSim-backed |
| `scripts/sac_fetchpush_coppeliasim.py` / `ddpg_fetchpush_coppeliasim.py` | Minimal-diff copies of the MuJoCo training scripts |
| `scripts/evaluate_policy_coppeliasim.py` | Minimal-diff copy of the MuJoCo eval script |
| `verify_custom_env_coppeliasim.py` | Sanity check + throughput probe |
| `scripts/coppeliasim/requirements_coppeliasim.txt` | Extra pip deps for this variant |
| `scripts/coppeliasim/convert_urdf_paths.py` | One-off xacro/mesh-path fixup for URDF import |
| `scripts/coppeliasim/launch_headless.py` | Optional headless CoppeliaSim launcher |
| `scripts/coppeliasim/assets/` | Vendored URDF/meshes + saved `.ttt` scene (untracked, local-only) |
