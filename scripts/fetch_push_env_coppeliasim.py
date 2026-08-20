"""
FetchPush on CoppeliaSim — experimental, unverified backend.

See docs/08-coppeliasim-variant.md for the full setup story. In short: this connects to an
*already-running* CoppeliaSim instance (with the scene from that doc built and loaded) via
the ZeroMQ Remote API, and produces the same (31,) observation / (4,) action / reward-type
contract as scripts/fetch_push_env.py's FetchPushFlatWrapper — so the same HERReplayBuffer,
and minimal-diff copies of the SAC/DDPG training scripts, work against it unchanged.

Nothing here has been run against a live CoppeliaSim instance (that requires the manually
built scene from the docs, which only you can create). Lines marked `# VERIFY:` are the ones
most likely to need adjustment once you can actually test against your scene/CoppeliaSim
version/physics engine.

Usage:
    from scripts.fetch_push_env_coppeliasim import register_fetch_push_coppeliasim_envs
    register_fetch_push_coppeliasim_envs()

    import gymnasium as gym
    env = gym.make("FetchPushCoppeliaSim-v0", reward_type="sparse")
"""

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Reused, not copied: pure math on (achieved_goal, desired_goal), zero sim dependency.
# Resolved with a fallback since this module is imported in two different sys.path contexts:
# bare (from `python scripts/sac_fetchpush_coppeliasim.py`, sys.path[0] == scripts/) and
# qualified (from `scripts.fetch_push_env_coppeliasim`, e.g. verify_custom_env_coppeliasim.py
# at the repo root, sys.path[0] == repo root).
try:
    from fetch_push_env import FetchPushFlatWrapper
except ImportError:
    from scripts.fetch_push_env import FetchPushFlatWrapper

# ---------------------------------------------------------------------------
# Object/joint paths in the CoppeliaSim scene. These depend entirely on what
# your URDF import actually produced — check the scene hierarchy after import
# and edit this dict to match. See docs/08-coppeliasim-variant.md step 8.
# ---------------------------------------------------------------------------
OBJECT_PATHS = {
    "arm_joints": [
        "/Fetch/shoulder_pan_joint",
        "/Fetch/shoulder_lift_joint",
        "/Fetch/upperarm_roll_joint",
        "/Fetch/elbow_flex_joint",
        "/Fetch/forearm_roll_joint",
        "/Fetch/wrist_flex_joint",
        "/Fetch/wrist_roll_joint",
    ],
    "finger_joints": [
        "/Fetch/l_gripper_finger_joint",
        "/Fetch/r_gripper_finger_joint",
    ],
    "gripper_tip": "/Fetch/IKTip",
    "ik_target": "/IKTarget",
    "ik_group_name": "FetchIK",  # VERIFY: name of the IK group as created in the IK plugin dialog
    "push_object": "/PushObject",
    "goal_marker": "/GoalMarker",
}

# Workspace bounds for sampling the object's initial position and the goal position, relative
# to the gripper's home position captured at first reset. Tune these to your actual scene's
# table dimensions — these are placeholders, not measured values.
OBJECT_XY_RANGE = np.array([0.15, 0.15])  # +/- meters around home, in the table plane
GOAL_XY_RANGE = np.array([0.15, 0.15])

ACTION_POS_SCALE = 0.05  # meters per unit action, matching the MuJoCo original's action[:3]*0.05
FINGER_CLOSED_POS = 0.0  # VERIFY: closed-position value for the finger joints on this gripper


class FetchPushCoppeliaSimEnv(gym.Env):
    """
    CoppeliaSim-backed equivalent of FetchPushFlatWrapper.

    Observation (31,): [0:25] robot_obs, [25:28] desired_goal, [28:31] achieved_goal — same
    layout as the MuJoCo wrapper, so HERReplayBuffer and the training scripts work unchanged.

    Action (4,): [dx, dy, dz] end-effector position delta (scaled by ACTION_POS_SCALE) applied
    via an IK target dummy; [3] is ignored (gripper is held closed, matching the MuJoCo
    original's block_gripper=True for the push task).
    """

    metadata = {"render_modes": [], "render_fps": 25}  # rgb_array not implemented yet, see docs

    REWARD_TYPES = FetchPushFlatWrapper.REWARD_TYPES

    def __init__(
        self,
        reward_type="sparse",
        host="localhost",
        port=23000,
        substeps_per_step=8,
        full_reset_every_n_episodes=50,
        success_threshold=0.05,
        randomize=False,
        object_mass_multiplier=1.0,
        friction_multiplier=1.0,
        object_size_multiplier=1.0,
        mass_range=None,
        friction_range=None,
        size_range=None,
    ):
        super().__init__()
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        except ImportError as e:
            raise ImportError(
                "coppeliasim-zmqremoteapi-client is required for the CoppeliaSim variant. "
                "Install with: pip install -r scripts/coppeliasim/requirements_coppeliasim.txt"
            ) from e

        self.reward_type = reward_type
        self.success_threshold = success_threshold
        self.substeps_per_step = substeps_per_step
        self.full_reset_every_n_episodes = full_reset_every_n_episodes
        self._episode_count = 0

        self.randomize = randomize
        self.object_mass_multiplier = object_mass_multiplier
        self.friction_multiplier = friction_multiplier
        self.object_size_multiplier = object_size_multiplier
        self.mass_range = mass_range or [1.0, 1.0]
        self.friction_range = friction_range or [1.0, 1.0]
        self.size_range = size_range or [1.0, 1.0]

        # VERIFY: host/port match your CoppeliaSim instance's ZMQ remote API server settings
        # (defaults are usually fine for a single local instance).
        self._client = RemoteAPIClient(host=host, port=port)
        self.sim = self._client.require("sim")
        self.simIK = self._client.require("simIK")

        self._resolve_handles()
        self.sim.setStepping(True)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float64)

        self._nominal_object_mass = None
        self._nominal_object_size = None
        self._home_gripper_pos = None
        self._home_ik_target_quat = None
        self._prev_distance = None
        self._prev_obs = None
        self._step_count = 0

        if self.sim.getSimulationState() == self.sim.simulation_stopped:
            self.sim.startSimulation()

    def _resolve_handles(self):
        sim = self.sim
        self._arm_joint_handles = [sim.getObject(p) for p in OBJECT_PATHS["arm_joints"]]
        self._finger_joint_handles = [sim.getObject(p) for p in OBJECT_PATHS["finger_joints"]]
        self._gripper_tip_handle = sim.getObject(OBJECT_PATHS["gripper_tip"])
        self._ik_target_handle = sim.getObject(OBJECT_PATHS["ik_target"])
        self._push_object_handle = sim.getObject(OBJECT_PATHS["push_object"])
        self._goal_marker_handle = sim.getObject(OBJECT_PATHS["goal_marker"])

        # VERIFY: this assumes the IK group was created in the scene via the IK plugin dialog
        # (docs/08-coppeliasim-variant.md step 6) and is retrievable by name. Newer CoppeliaSim
        # versions (4.5+) may prefer building the IK environment programmatically via
        # simIK.createIkEnvironment()/createIkGroup() instead — if getIkGroupHandle fails with
        # your installed version, that's the alternative to switch to.
        self._ik_group_handle = self.simIK.getIkGroupHandle(OBJECT_PATHS["ik_group_name"])

        self._sim_dt = self.sim.getFloatParam(self.sim.floatparam_simulation_time_step)

    # -----------------------------------------------------------------
    # Domain randomization (the two genuinely simulator-specific methods —
    # everything else in this class besides these mirrors the MuJoCo wrapper's structure).
    # -----------------------------------------------------------------

    def _save_nominal_params_sim(self):
        if self._nominal_object_mass is not None:
            return
        self._nominal_object_mass = self.sim.getShapeMass(self._push_object_handle)
        # Size isn't directly queryable as a scalar the way MuJoCo's geom_size is; track our
        # own applied-scale state instead of reading it back from CoppeliaSim.
        self._nominal_size_scale = 1.0

    def _apply_domain_randomization_sim(self):
        if self._nominal_object_mass is None:
            return

        if self.randomize:
            mass_mult = np.random.uniform(*self.mass_range)
            friction_mult = np.random.uniform(*self.friction_range)
            size_mult = np.random.uniform(*self.size_range)
        else:
            mass_mult = self.object_mass_multiplier
            friction_mult = self.friction_multiplier
            size_mult = self.object_size_multiplier

        try:
            self.sim.setShapeMass(self._push_object_handle, self._nominal_object_mass * mass_mult)
        except Exception:
            pass  # matches the MuJoCo wrapper's graceful-degradation behavior

        # VERIFY: friction parameter name/enum depends on which physics engine your scene uses
        # (Bullet/ODE/Newton/Vortex) — could not be confirmed without a running instance. Try
        # something like sim.setEngineFloatParam(sim.bullet_body_friction, self._push_object_handle,
        # nominal_friction * friction_mult) and adjust the constant name for your engine.
        try:
            pass  # intentionally not implemented — see VERIFY note above
        except Exception:
            pass

        try:
            relative_scale = (size_mult) / self._nominal_size_scale
            self.sim.scaleObject(self._push_object_handle, relative_scale, relative_scale, relative_scale)
            self._nominal_size_scale = size_mult
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Observation construction — mirrors the 25-dim layout from
    # gymnasium_robotics' fetch_env.py: [grip_pos(3), object_pos(3), object_rel_pos(3),
    # gripper_state(2), object_rot(3), object_velp(3), object_velr(3), grip_velp(3), gripper_vel(2)]
    # -----------------------------------------------------------------

    def _get_obs(self):
        sim = self.sim
        dt = self._sim_dt * self.substeps_per_step

        grip_pos = np.array(sim.getObjectPosition(self._gripper_tip_handle, -1))
        object_pos = np.array(sim.getObjectPosition(self._push_object_handle, -1))
        object_rel_pos = object_pos - grip_pos

        finger_positions = np.array([sim.getJointPosition(h) for h in self._finger_joint_handles])
        gripper_state = finger_positions  # (2,)

        object_rot = np.array(sim.getObjectOrientation(self._push_object_handle, -1))  # Euler, (3,)

        grip_lin_vel, grip_ang_vel = sim.getObjectVelocity(self._gripper_tip_handle)
        object_lin_vel, object_ang_vel = sim.getObjectVelocity(self._push_object_handle)

        grip_velp = np.array(grip_lin_vel) * dt
        object_velp = (np.array(object_lin_vel) - np.array(grip_lin_vel)) * dt
        object_velr = np.array(object_ang_vel) * dt

        finger_velocities = np.array([sim.getJointVelocity(h) for h in self._finger_joint_handles])
        gripper_vel = finger_velocities * dt

        robot_obs = np.concatenate([
            grip_pos, object_pos, object_rel_pos, gripper_state,
            object_rot, object_velp, object_velr, grip_velp, gripper_vel,
        ])  # (25,)

        achieved_goal = object_pos.copy()  # (3,)
        desired_goal = np.array(sim.getObjectPosition(self._goal_marker_handle, -1))  # (3,)

        flat_obs = np.concatenate([robot_obs, desired_goal, achieved_goal])
        return flat_obs, achieved_goal, desired_goal

    # -----------------------------------------------------------------
    # Reward — ported 1:1 from FetchPushFlatWrapper._compute_reward's formulas (not imported,
    # since that method is an instance method reading self._prev_distance/_step_count and
    # isn't meant as a cross-class API). Keep this in sync by hand if the original changes.
    # -----------------------------------------------------------------

    def _compute_reward(self, achieved_goal, desired_goal, action, distance, is_success):
        if self.reward_type == "sparse":
            return 0.0 if is_success else -1.0
        elif self.reward_type == "dense_basic":
            return -distance
        elif self.reward_type == "progress_bonus":
            sparse_base = 0.0 if is_success else -1.0
            progress = self._prev_distance - distance
            progress_reward = 10.0 * progress
            success_bonus = 5.0 if is_success else 0.0
            return sparse_base + progress_reward + success_bonus
        elif self.reward_type == "energy_efficient":
            dense_distance = -distance
            energy_penalty = -0.1 * np.sum(action[:3] ** 2)
            success_bonus = 1.0 if is_success else 0.0
            return dense_distance + energy_penalty + success_bonus
        else:
            raise ValueError(
                f"Unknown or unsupported reward_type for CoppeliaSim variant: '{self.reward_type}'. "
                f"multi_component is not ported here (same HER-incompatibility as the MuJoCo original)."
            )

    # -----------------------------------------------------------------
    # Action application — IK target delta + finger lock + physics substeps,
    # analogous to the MuJoCo original's mocap-weld + block_gripper mechanism.
    # -----------------------------------------------------------------

    def _apply_action_and_step(self, action):
        sim, simIK = self.sim, self.simIK
        delta = np.asarray(action[:3], dtype=np.float64) * ACTION_POS_SCALE

        current_target_pos = np.array(sim.getObjectPosition(self._ik_target_handle, -1))
        new_target_pos = current_target_pos + delta
        sim.setObjectPosition(self._ik_target_handle, -1, new_target_pos.tolist())
        # Orientation held fixed at whatever was captured at reset (scene-specific, not a
        # hardcoded MuJoCo quaternion — engines use different quaternion/axis conventions).
        if self._home_ik_target_quat is not None:
            sim.setObjectQuaternion(self._ik_target_handle, -1, self._home_ik_target_quat)

        simIK.handleIkGroup(self._ik_group_handle)

        for h in self._finger_joint_handles:
            sim.setJointTargetPosition(h, FINGER_CLOSED_POS)

        for _ in range(self.substeps_per_step):
            self._client.step()

    # -----------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._save_nominal_params_sim()
        self._apply_domain_randomization_sim()

        do_full_reset = (
            self.full_reset_every_n_episodes > 0
            and self._episode_count % self.full_reset_every_n_episodes == 0
            and self._episode_count > 0
        )
        if do_full_reset:
            self.sim.stopSimulation()
            while self.sim.getSimulationState() != self.sim.simulation_stopped:
                time.sleep(0.05)
            self.sim.startSimulation()

        if self._home_gripper_pos is None:
            # Captured once, empirically, from wherever URDF import actually placed the robot —
            # not assumed to match MuJoCo's hardcoded offset (see docs, "coordinate frame" risk).
            self._home_gripper_pos = np.array(self.sim.getObjectPosition(self._gripper_tip_handle, -1))
            self._home_ik_target_quat = self.sim.getObjectQuaternion(self._ik_target_handle, -1)

        object_xy_offset = self.np_random.uniform(-OBJECT_XY_RANGE, OBJECT_XY_RANGE)
        goal_xy_offset = self.np_random.uniform(-GOAL_XY_RANGE, GOAL_XY_RANGE)
        object_pos = self._home_gripper_pos.copy()
        object_pos[:2] += object_xy_offset
        goal_pos = self._home_gripper_pos.copy()
        goal_pos[:2] += goal_xy_offset

        self.sim.resetDynamicObject(self._push_object_handle)
        self.sim.setObjectPosition(self._push_object_handle, -1, object_pos.tolist())
        self.sim.setObjectPosition(self._goal_marker_handle, -1, goal_pos.tolist())
        self.sim.setObjectPosition(self._ik_target_handle, -1, self._home_gripper_pos.tolist())

        for h in self._arm_joint_handles + self._finger_joint_handles:
            self.sim.resetDynamicObject(h)

        for _ in range(10):
            self._client.step()

        flat_obs, achieved_goal, desired_goal = self._get_obs()
        self._prev_distance = float(np.linalg.norm(achieved_goal - desired_goal))
        self._prev_obs = flat_obs
        self._step_count = 0
        self._episode_count += 1

        info = {
            "is_success": False,
            "desired_goal": desired_goal,
            "achieved_goal": achieved_goal,
        }
        return flat_obs, info

    def step(self, action):
        self._apply_action_and_step(action)
        self._step_count += 1

        flat_obs, achieved_goal, desired_goal = self._get_obs()
        distance = float(np.linalg.norm(achieved_goal - desired_goal))
        is_success = distance < self.success_threshold

        reward = self._compute_reward(achieved_goal, desired_goal, action, distance, is_success)

        self._prev_distance = distance
        self._prev_obs = flat_obs

        terminated = False  # matches the MuJoCo original: success doesn't end the episode early
        truncated = False  # max_episode_steps is enforced by the gym.register() TimeLimit wrapper

        info = {
            "is_success": is_success,
            "distance_to_goal": distance,
            "desired_goal": desired_goal,
            "achieved_goal": achieved_goal,
        }
        return flat_obs, reward, terminated, truncated, info

    def close(self):
        pass  # deliberately does not stop the CoppeliaSim simulation — see docs, "running instance model"

    @staticmethod
    def compute_reward_static(achieved_goal, desired_goal, reward_type="sparse", threshold=0.05):
        return FetchPushFlatWrapper.compute_reward_static(achieved_goal, desired_goal, reward_type, threshold)

    @staticmethod
    def get_goal_from_obs(obs, goal_dim=3):
        return FetchPushFlatWrapper.get_goal_from_obs(obs, goal_dim)


def register_fetch_push_coppeliasim_envs():
    """Register the CoppeliaSim-backed FetchPush environment with Gymnasium."""
    gym.register(
        id="FetchPushCoppeliaSim-v0",
        entry_point=FetchPushCoppeliaSimEnv,
        kwargs={"reward_type": "sparse"},
        max_episode_steps=50,
    )


if __name__ == "__main__":
    register_fetch_push_coppeliasim_envs()
    print("Registered FetchPushCoppeliaSim-v0. Run verify_custom_env_coppeliasim.py to sanity-check "
          "against a running CoppeliaSim instance.")
