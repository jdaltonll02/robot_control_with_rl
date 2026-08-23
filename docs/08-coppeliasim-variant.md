# CoppeliaSim Variant (attempted, not completed — see notes below)

A second, independent backend for the same goal-conditioned FetchPush task, running on
CoppeliaSim (EDU) instead of MuJoCo, using the real Fetch robot's URDF. Nothing in the
original MuJoCo pipeline (`scripts/fetch_push_env.py`, `her_replay_buffer.py`,
`sac_fetchpush.py`, `ddpg_fetchpush.py`, `evaluate_policy.py`, `verify_custom_env.py`) is
modified — every file listed below is new.

**Status: attempted through a full scene build, did not reach a working state.** The scene
(URDF import, table/object/goal, IK group) was built by hand following the steps below, and
the Python module was written against it — but the resulting scene has a real physics
instability (non-convex dynamic collision shapes → severe slowdown, and a failed convex-hull
fix that caused a physics explosion instead of resolving it). **See
[09-alternative-simulators.md](09-alternative-simulators.md) for the full account of what was
tried, what broke, and why** — this project ultimately settled on the working MuJoCo pipeline
as the sole delivered environment. The setup steps below are left intact as accurate
instructions for the parts that did work (URDF import, object/dummy creation, IK wizard
usage); only the final collision-shape step is the known broken point.

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
3. **Import into CoppeliaSim:** `Modules → Importers → URDF importer...` → select
   `fetch_fixed.urdf` (menu path confirmed against CoppeliaSim EDU 4.3+ — older docs/tutorials
   may say "Plugins", that's an outdated menu name). This produces a tree of link/joint
   objects for the real 7-DOF Fetch arm + torso + gripper, named after the URDF's own joint
   names (`shoulder_pan_joint`, `elbow_flex_joint`, `l_gripper_finger_joint`, etc.), each
   nested under a `*_link_respondable` object (the physical/collision body) with a
   `*_link_visual` child (the pure appearance mesh) — this is exactly what `OBJECT_PATHS`
   in `fetch_push_env_coppeliasim.py` is already written against, confirmed against a real
   import. There's no top-level "Fetch" grouping object — the chain starts directly at
   `base_link_respondable`.

4. **Add a table.** The URDF import only brings in the robot — right now it would be
   standing on bare floor with nothing to push things across. `Add → Shape → Primitive
   shape → Cuboid`, size it flat and roughly table-height (e.g. 0.6m × 0.6m × 0.05m, top
   surface somewhere around the height the gripper naturally hangs at when the arm is in a
   neutral pose — eyeball it against the rendered robot). Rename it `Table` (double-click its
   name in the Scene hierarchy, or right-click → Rename). Leave "Body is dynamic" **unchecked**
   (it shouldn't fall or move) but leave "Body is respondable" **checked** (so things can
   rest on it and the robot can't pass through it) — these two checkboxes are on the shape's
   properties dialog, opened by double-clicking the shape in the 3D view or in the Scene
   hierarchy.

5. **Add `PushObject`.** `Add → Shape → Primitive shape → Cuboid` again, this time small
   (a few cm per side), positioned sitting on top of the table. Rename it `PushObject`. Open
   its properties dialog and check **both** "Body is dynamic" (gravity/physics affects it —
   this is the thing that gets pushed) **and** "Body is respondable" (so the gripper and table
   can actually collide with it instead of passing through).

6. **Add `GoalMarker`.** `Add → Shape → Primitive shape → Sphere` (or cuboid), small, also on
   the table, some distance from `PushObject`. Rename it `GoalMarker`. Leave **both** "Body is
   dynamic" and "Body is respondable" **unchecked** — this is a pure visual marker for where
   the object should end up, it must never physically interact with anything (a respondable
   goal marker would itself become an obstacle the object bumps into, which isn't the task).
   Optionally give it a distinct color (right-click → Edit → Shape color, or the color
   toolbar icon) so it's easy to see in a screenshot.

7. **Set up the IK group** (the CoppeliaSim analog of MuJoCo's mocap-weld mechanism):
   - `Add → Dummy` — this creates a small axis-marker object. Rename it `IKTip`.
     **Parent it to the gripper** by dragging `IKTip` onto `gripper_link_respondable` in the
     Scene hierarchy tree (drag-and-drop reparents it). After reparenting, open `IKTip`'s
     properties dialog and set its **position relative to its parent** to `(0, 0, 0)` (or a
     small forward offset if you want the IK tip to sit between the two gripper fingers
     rather than at the wrist) — this makes it move rigidly with the gripper from now on,
     exactly like the point MuJoCo's mocap weld attaches to.
   - `Add → Dummy` again for the second one — rename it `IKTarget`. Leave it **unparented**
     (directly under the scene root) — this is the one the Python side repositions every
     training step. Before you move on, set its starting position to match `IKTip`'s current
     *world* position (open both dummies' properties dialogs and copy the world X/Y/Z from
     one to the other, or use `Edit → Align to...` with `IKTip` as reference) — starting them
     coincident avoids a large, jarring IK jump the moment the simulation starts.
   - `Modules → Kinematics → Inverse kinematics generator...` — confirmed present in your
     CoppeliaSim EDU build. This is a **wizard**, not a raw IK-group dialog: it generates a
     self-running child script attached to the robot model that resolves IK automatically
     every simulation step, rather than something the Python side has to trigger manually.
     Fill in: **Robot model** = the Fetch model (should be the only option), **Robot base** =
     `torso_lift_link_respondable` (excludes torso-lift/wheel joints from the solved chain —
     the original task doesn't move the torso either), **Robot tip** = `IKTip`, **Robot
     target** = `IKTarget`, **Joint group** = leave empty ("use all joints in the tip-target
     chain" is exactly right — that's the 7 arm joints). Leave Position X/Y/Z and Orientation
     Alpha+Beta/Gamma all checked (we want the fixed end-effector orientation constrained too,
     matching MuJoCo). Leave the solver defaults (10 iterations, 0.1 damping) for now — this is
     exactly the kind of thing flagged below for later tuning. Under Handling, keep "Allow
     error" and "Enabled"/"During simulation" checked — "Allow error" matters: without it, the
     script may freeze the arm entirely rather than best-effort-solving when the target is
     briefly unreachable. Click Generate.
   - Because this wizard's script runs automatically every simulation step (as long as
     "During simulation" is checked), the Python wrapper does **not** call any `simIK`
     function directly — it only moves `IKTarget`'s position/orientation and then calls
     `client.step()`, and the scene's own generated script does the rest. This is actually a
     *closer* match to MuJoCo's continuously-resolved mocap-weld than the originally-planned
     "one manual `simIK.handleIkGroup()` call per env step" would have been — updated in
     `fetch_push_env_coppeliasim.py` accordingly.
   - **Fidelity note**: MuJoCo's mocap+weld is resolved *compliantly* by the physics solver
     every one of its 20 internal substeps per control step. The wizard-generated script
     resolves similarly — once per simulation step, not once per env step — via the arm's
     **position-controlled dynamic joints**, letting the engine's PD servo converge as physics
     advances. The solver's iteration count/damping factor (set in the wizard above) are the
     most likely things to need hand-tuning once you can actually run it.

8. **Save the scene.** `Scenes → Save scene as...` (a top-level menu, separate from `File`,
   confirmed in your version's menu bar) → save as
   `scripts/coppeliasim/assets/fetch_push_scene.ttt` (also untracked, like the URDF assets —
   binary, local-only). If your version doesn't have a `Scenes` menu, use `File → Save Scene
   As...` instead.

9. **Confirm/update `OBJECT_PATHS`** at the top of `scripts/fetch_push_env_coppeliasim.py`.
    The arm/finger joint names are already updated to match a real import (no `/Fetch/`
    prefix — CoppeliaSim resolves `"/name"` by unique alias regardless of nesting, so
    `/IKTip` still resolves correctly even though it's parented under
    `gripper_link_respondable`). Just double check `push_object`, `goal_marker`,
    `gripper_tip`, and `ik_target` match exactly what you named things above — that dict is
    the *only* place these names live in the code. (There's no `ik_group_name` entry anymore
    — the IK wizard's generated script doesn't need to be looked up from Python.)

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
