# Alternative Simulators — Investigation Notes (not completed)

After the main project goal was achieved (SAC+HER on Gymnasium-Robotics' MuJoCo-based
`FetchPush-v4` — see the [README's Results](../README.md#results)), three alternative
simulator backends were explored out of curiosity about running the same task against the
*real* Fetch robot's URDF, or on a different physics engine entirely. **None reached a
working, trainable state.** This doc records what was tried and why, so the investigation
isn't silently lost and nobody re-discovers the same dead ends from scratch.

The working MuJoCo pipeline was never touched during any of this — it remains the sole
delivered environment.

## Summary

| Attempt | Real URDF loads? | Stable dynamics? | Precise Cartesian control? |
|---|---|---|---|
| CoppeliaSim | Only after manual GUI convex-decomposition work | **No** — physics explosion | Never reached |
| MuJoCo (raw URDF, new scene) | Yes, after 3 manual fixes | **No** — weld drifted or exploded | Never reached |
| PyBullet | Yes, zero preprocessing | **Yes** — never diverged | **No** — stable but wrong (gravity sag) |

The pattern across all three: **driving a real, uncalibrated URDF via Cartesian/IK control is
a genuine control-engineering problem, independent of simulator.** Gymnasium-Robotics' own
Fetch MJCF (the one behind all of this project's actual results) works well specifically
because someone already did that tuning work — using the raw public URDF starts from zero.

## 1. CoppeliaSim (see [08-coppeliasim-variant.md](08-coppeliasim-variant.md) for full setup)

Got furthest on setup effort: real Fetch URDF imported via CoppeliaSim's GUI, a full scene
built by hand (table, `PushObject`, `GoalMarker`, `IKTip`/`IKTarget` dummies, an IK group via
the "Inverse Kinematics generator" wizard), and a complete Python-side module
(`scripts/fetch_push_env_coppeliasim.py` + minimal-diff training/eval script variants) written
against that scene's expected object names.

**What went wrong:** the imported robot's dynamic collision shapes used the raw (non-convex)
visual meshes, which CoppeliaSim explicitly warns is unstable for a physics engine. Running
`verify_custom_env_coppeliasim.py` measured **0.9 steps/sec** (a 250K-step run would take
75+ hours) and CoppeliaSim's console warned: *"Detected dynamically enabled, non-convex
shapes. Those might drastically slow down simulation, and introduce unstable behaviour."*

Attempting the fix (V-HACD convex decomposition via `Modules → Geometry/Mesh → Convex
decomposition`) made it *worse*, not better: the decomposition tool added new convex hull
shapes *alongside* the original non-convex ones rather than replacing them, so two dynamic
rigid bodies ended up representing the same physical link, interpenetrating and violently
flinging the model apart (one link was measured at `z=-117`, a clear physics explosion). The
scene was saved in this broken state.

**Status:** the Python module code is real and reusable, but the scene needs to be rebuilt
from scratch with correct convex-hull collision assignment (deleting/disabling the original
non-convex shape's dynamics after decomposition, not just adding new hulls next to it) before
any of it can run.

## 2. MuJoCo with the raw Fetch URDF (not the same as the working pipeline)

Idea: since MuJoCo already works reliably in this project, load the *actual* Fetch URDF
directly into it (`mujoco.MjModel.from_xml_path`) instead of relying on Gymnasium-Robotics'
existing Fetch model, to get real-robot fidelity without leaving a known-good simulator.

**Getting the model to load required three real, concrete fixes:**
1. MuJoCo's URDF importer discards absolute mesh paths and re-resolves by basename via its
   own `meshdir` — fixed with a `<mujoco><compiler meshdir="..." strippath="true"/></mujoco>`
   extension block (a documented MuJoCo-specific URDF augmentation).
2. MuJoCo can't decode `.dae` (COLLADA) meshes, only STL/OBJ/MSH — fixed with
   `discardvisual="true"`, falling back to the STL collision meshes.
3. Both gripper-finger links had genuinely degenerate inertia tensors in the public URDF
   itself (`iyy="0" izz="0"`, not just poorly conditioned) — patched to isotropic values.

Once loaded (confirmed: 16 bodies, 15 joints, all named correctly), a full task scene was
hand-built around it: table, `object0` (free-joint box), `target0` site, a `mocap` body +
weld equality constraint to a new `gripper_tip` body, and position actuators for the gripper
fingers.

**What went wrong:** the weld constraint couldn't hold a commanded arm pose reliably.
- At default weld stiffness, the arm's tip drifted ~9cm away from its held position over a
  300-step settle with the mocap target completely fixed. Two hypotheses were tested and
  both were **disproven**: it wasn't a position/orientation conflict (checked directly — the
  orientations matched), and it wasn't pure gravity sag either (disabling the weld entirely
  produced drift in a different direction than what the weld-enabled case showed). The exact
  mechanism was not identified.
- Stiffening the weld to compensate caused a genuine numerical explosion
  (`max|qvel|` spiked to ~1108, the object was flung to `(1.84, 0.84, 0.79)`), a classic
  constraint-stiffness-vs-timestep instability.

**Status:** the loading fixes are real, reusable knowledge (documented above) if anyone wants
to load this URDF into MuJoCo again. The task scene and weld mechanism need to be rebuilt with
either explicit per-joint position actuators (not relying on a bare weld) or much more careful
solver/timestep tuning.

## 3. PyBullet

The most promising of the three, tried last after both other engines hit instability.

**What worked immediately, with zero preprocessing:** `pybullet.loadURDF()` loaded the
**completely unmodified** original public URDF directly — no mesh-path fixes, no `.dae`
conversion, no inertia patching. All 25 joints parsed correctly by name, all 21 visual
meshes (mixing `.dae` and `.stl`) loaded with correct geometry (verified via a sane,
non-degenerate AABB). `pybullet.calculateInverseKinematics()` — a single function call —
never produced instability in any test: no explosions, no NaN, no divergence, across dozens
of trials.

**What didn't get solved:** precise Cartesian tracking. A plain `POSITION_CONTROL` loop
converged to a **stable** but **wrong** steady-state — a consistent ~9cm offset from the
commanded target that didn't change across 50, 100, or 200 simulation steps. This is a
textbook symptom of gravity sag under a PD controller with no gravity compensation. An
attempted fix using `calculateInverseDynamics()` for explicit gravity compensation made
things *worse* (error jumped to 0.5–0.8m, oscillating) — very likely a DOF-index mapping bug
in the quick hand-rolled controller (the assumption that `movable_indices` order matches
`calculateInverseDynamics`'s internal DOF order was never verified), not a PyBullet
limitation.

**Status:** no environment module was written for this attempt — it stayed at the raw
feasibility-testing stage (no `fetch_push_env_pybullet.py` exists). If picked up again, the
next concrete step is verifying the DOF-index mapping assumption before re-attempting gravity
compensation, or trying PyBullet's own higher-level tools (e.g. increasing `positionGain`
substantially on the plain position-control path, which was never pushed past PyBullet's
suggested defaults) before hand-rolling torque control again.

## If picked back up later

PyBullet is the recommended starting point of the three — it's the only one that never
diverged. The concrete unblocking step is fixing (or replacing) the gravity-compensation
controller, then following the same pattern used for the CoppeliaSim variant: a new
`fetch_push_env_pybullet.py` implementing the same `(31,)`/`(4,)` contract described in
[architecture.md](architecture.md), reusing `HERReplayBuffer` and the reward math unmodified,
with minimal-diff copies of the training/eval scripts.
