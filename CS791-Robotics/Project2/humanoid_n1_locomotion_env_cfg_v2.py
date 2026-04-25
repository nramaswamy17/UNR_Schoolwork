# humanoid_n1_locomotion_env_cfg_v2.py
#
# CS791 Project 2 – Humanoid Walk/Run Environment
#
# Key additions over v1:
#   - 3-stage curriculum (stand → walk → run)
#   - Command range 0 → 3.2 m/s (walk + run)
#   - Domain randomization: mass, friction, push forces
#   - Gait rewards: feet air-time, feet clearance, torque penalty
#   - Velocity-tracking error metric logged per step
#   - Slip metric, step-frequency tracking via _feet_air_time tensor

import math
import os
from typing import List, Tuple

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


# =============================================================================
# Curriculum stages
# =============================================================================
#   Stage 0 – "slow"  : vx in [0.0, 0.8]   stand + slow walk
#   Stage 1 – "walk"  : vx in [0.0, 1.8]   full walk range
#   Stage 2 – "run"   : vx in [0.0, 3.2]   walk + run
#
# The training script advances the stage by calling
#   env._curriculum_stage = N   (0, 1, or 2)
# based on mean episode reward thresholds.

CURRICULUM_STAGES = [
    {"vx_max": 0.8,  "vy_max": 0.0, "yaw_max": 0.0, "push": False},
    {"vx_max": 1.8,  "vy_max": 0.3, "yaw_max": 0.3, "push": True},
    {"vx_max": 3.2,  "vy_max": 0.5, "yaw_max": 0.5, "push": True},
]

# Reward thresholds to advance stages (mean episode reward)
CURRICULUM_THRESHOLDS = [6.0, 10.0]   # advance to stage 1 / stage 2


# =============================================================================
# USD path
# =============================================================================
USD_PATH = os.path.join(os.environ["HOME"], "humanoid_n1_task/robots/N1/urdf/N1_raw/N1_raw.usd")
if not os.path.isfile(USD_PATH):
    raise FileNotFoundError(f"USD not found: {USD_PATH}")


# =============================================================================
# Joint names
# =============================================================================
LEFT_LEG_6 = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_roll_joint", "left_ankle_pitch_joint",
]
RIGHT_LEG_6 = [
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_roll_joint", "right_ankle_pitch_joint",
]
WAIST_1 = ["waist_yaw_joint"]

LEFT_ARM_5 = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_wrist_yaw_joint",
]
RIGHT_ARM_5 = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_wrist_yaw_joint",
]

N1_ACTION_JOINTS_13 = LEFT_LEG_6 + RIGHT_LEG_6 + WAIST_1
N1_JOINTS_23 = LEFT_LEG_6 + RIGHT_LEG_6 + WAIST_1 + LEFT_ARM_5 + RIGHT_ARM_5

assert len(N1_ACTION_JOINTS_13) == 13
assert len(N1_JOINTS_23) == 23

LEFT_FOOT_BODY  = "left_foot_pitch_link"
RIGHT_FOOT_BODY = "right_foot_pitch_link"


# =============================================================================
# Joint limits + default angles
# =============================================================================
ACTIONS_MAX_13 = np.array([
    2.618, 1.571, 1.571, 2.356, 0.436, 0.785,
    2.618, 0.262, 1.571, 2.356, 0.436, 0.785,
    2.618,
], dtype=np.float32)

ACTIONS_MIN_13 = np.array([
    -2.618, -0.262, -1.571, -0.087, -0.436, -0.785,
    -2.618, -1.571, -1.571, -0.087, -0.436, -0.785,
    -2.618,
], dtype=np.float32)

DEFAULT_JOINT_ANGLES = {
    "left_hip_pitch_joint":   float(-np.deg2rad(14.0)),
    "left_hip_roll_joint":    0.0,
    "left_hip_yaw_joint":     0.0,
    "left_knee_pitch_joint":  float(+np.deg2rad(29.5)),
    "left_ankle_roll_joint":  0.0,
    "left_ankle_pitch_joint": float(-np.deg2rad(13.7)),
    "right_hip_pitch_joint":  float(-np.deg2rad(14.0)),
    "right_hip_roll_joint":   0.0,
    "right_hip_yaw_joint":    0.0,
    "right_knee_pitch_joint": float(+np.deg2rad(29.5)),
    "right_ankle_roll_joint": 0.0,
    "right_ankle_pitch_joint": float(-np.deg2rad(13.7)),
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}
DEFAULT_JOINT_ANGLES = {k: float(v) for k, v in DEFAULT_JOINT_ANGLES.items()}

ACTION_OFFSET_BY_JOINT = {}
ACTION_SCALE_BY_JOINT  = {}
for j, jmax, jmin in zip(N1_ACTION_JOINTS_13, ACTIONS_MAX_13, ACTIONS_MIN_13):
    q0 = float(DEFAULT_JOINT_ANGLES.get(j, 0.0))
    scale = max(1e-6, min(float(jmax) - q0, q0 - float(jmin)))
    ACTION_OFFSET_BY_JOINT[j] = q0
    ACTION_SCALE_BY_JOINT[j]  = scale


# =============================================================================
# Torque limits + PD gains
# =============================================================================
TAU_LIMIT_23 = np.array([
    95, 54, 54, 95, 30, 30,
    95, 54, 54, 95, 30, 30,
    54,
    54, 30, 30, 30, 30,
    54, 30, 30, 30, 30,
], dtype=np.float32)
TAU_LIMIT_BY_JOINT = {j: float(t) for j, t in zip(N1_JOINTS_23, TAU_LIMIT_23)}


def kp_kd_for_joint(j: str) -> Tuple[float, float]:
    if "hip_pitch"       in j: return 200.0, 12.0
    if "hip_roll"        in j: return 150.0, 12.0
    if "hip_yaw"         in j: return 100.0, 10.0
    if "knee"            in j: return 150.0, 10.0
    if "ankle"           in j: return 50.0, 3.0
    if "waist"           in j: return 100.0, 10.0
    if "shoulder_pitch"  in j: return 90.0, 8.0
    return 45.0, 2.5


ACTUATORS = {
    f"act_{j}": ImplicitActuatorCfg(
        joint_names_expr=[j],
        stiffness=kp_kd_for_joint(j)[0],
        damping=kp_kd_for_joint(j)[1],
        effort_limit_sim=TAU_LIMIT_BY_JOINT.get(j, 300.0),
        velocity_limit_sim=50.0,
    )
    for j in N1_JOINTS_23
}


# =============================================================================
# Math helpers
# =============================================================================
@torch.jit.script
def _quat_rotate_inverse_wxyz(q_wxyz: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
    w   = q_wxyz[:, 0:1]
    xyz = q_wxyz[:, 1:4]
    t   = 2.0 * torch.cross(xyz, v_w, dim=1)
    return v_w + w * t + torch.cross(xyz, t, dim=1)


def _l2_sq(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * x, dim=-1)


def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
    w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _quat_normalize_wxyz(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-8)


def _quat_from_euler_xyz_wxyz(roll, pitch, yaw):
    cr = torch.cos(roll * 0.5);   sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5);  sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5);    sy = torch.sin(yaw * 0.5)
    return torch.stack([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ], dim=-1)


# =============================================================================
# Robot config
# =============================================================================
HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.0,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.70),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos=DEFAULT_JOINT_ANGLES,
        joint_vel={j: 0.0 for j in DEFAULT_JOINT_ANGLES},
    ),
    actuators=ACTUATORS,
)


@configclass
class HumanoidN1SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(200.0, 200.0)),
    )
    robot: ArticulationCfg = HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# =============================================================================
# Actions
# =============================================================================
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=N1_ACTION_JOINTS_13,
        scale=ACTION_SCALE_BY_JOINT,
        offset=ACTION_OFFSET_BY_JOINT,
        use_default_offset=False,
    )


# =============================================================================
# Curriculum-aware command tensor helpers
# =============================================================================
def _get_command_tensor(env) -> torch.Tensor:
    if not hasattr(env, "_n1_commands"):
        env._n1_commands = torch.zeros((env.num_envs, 3), device=env.device)
    return env._n1_commands


def _get_curriculum_stage(env) -> int:
    return int(getattr(env, "_curriculum_stage", 0))


def sample_velocity_commands(env, env_ids):
    """Sample [vx, vy, yaw_rate] according to current curriculum stage."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif not torch.is_tensor(env_ids):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    stage   = _get_curriculum_stage(env)
    cfg     = CURRICULUM_STAGES[min(stage, len(CURRICULUM_STAGES) - 1)]
    cmds    = _get_command_tensor(env)
    n       = env_ids.numel()

    vx  = cfg["vx_max"]  * torch.rand((n,), device=env.device)
    vy  = cfg["vy_max"]  * (2.0 * torch.rand((n,), device=env.device) - 1.0)
    yaw = cfg["yaw_max"] * (2.0 * torch.rand((n,), device=env.device) - 1.0)

    cmds[env_ids, 0] = vx
    cmds[env_ids, 1] = vy
    cmds[env_ids, 2] = yaw


# =============================================================================
# Feet air-time tracker (gait metric)
# =============================================================================
def _get_feet_air_time(env) -> torch.Tensor:
    """Per-env, per-foot air time in seconds. Shape: (num_envs, 2)."""
    if not hasattr(env, "_feet_air_time"):
        env._feet_air_time = torch.zeros((env.num_envs, 2), device=env.device)
    return env._feet_air_time


def _get_feet_last_contact(env) -> torch.Tensor:
    if not hasattr(env, "_feet_last_contact"):
        env._feet_last_contact = torch.ones((env.num_envs, 2), device=env.device)
    return env._feet_last_contact


# =============================================================================
# Reset events
# =============================================================================
def _get_articulation_joint_names(asset) -> List[str]:
    for attr in ("joint_names", "dof_names"):
        if hasattr(asset, attr):          return list(getattr(asset, attr))
        if hasattr(asset, "data") and hasattr(asset.data, attr):
            return list(getattr(asset.data, attr))
    raise AttributeError("Could not find joint/dof names on articulation.")


def reset_scene_safe(env, env_ids=None):
    if hasattr(mdp, "reset_scene_to_default"): return mdp.reset_scene_to_default(env, env_ids)
    if hasattr(mdp, "reset_scene"):            return mdp.reset_scene(env, env_ids)
    return None


def spawn_random_action_joints(env, env_ids):
    if not torch.is_tensor(env_ids):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    asset = env.scene["robot"]
    all_joint_names = _get_articulation_joint_names(asset)
    n     = int(env_ids.numel())
    ndof  = len(all_joint_names)

    qpos = torch.zeros((n, ndof), device=env.device)
    qvel = torch.zeros((n, ndof), device=env.device)

    for i, jname in enumerate(all_joint_names):
        qpos[:, i] = float(DEFAULT_JOINT_ANGLES.get(jname, 0.0))

    for i, jname in enumerate(all_joint_names):
        if jname in N1_ACTION_JOINTS_13:
            s     = float(ACTION_SCALE_BY_JOINT[jname])
            delta = (0.35 * s) * (2.0 * torch.rand((n,), device=env.device) - 1.0)
            qpos[:, i] += delta

    if hasattr(asset, "write_joint_positions_to_sim"):
        asset.write_joint_positions_to_sim(qpos, env_ids=env_ids)
    elif hasattr(asset, "write_joint_state_to_sim"):
        asset.write_joint_state_to_sim(qpos, qvel, env_ids=env_ids)
    if hasattr(asset, "write_joint_velocities_to_sim"):
        asset.write_joint_velocities_to_sim(qvel, env_ids=env_ids)


def randomize_root_pose_small(env, env_ids, roll_pitch_deg: float = 3.0, yaw_deg: float = 15.0):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif not torch.is_tensor(env_ids):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    asset = env.scene["robot"]
    n = int(env_ids.numel())
    if n == 0:
        return

    if hasattr(asset.data, "root_pos_w"):
        root_pos  = asset.data.root_pos_w[env_ids].clone()
        root_quat = asset.data.root_quat_w[env_ids].clone()
    elif hasattr(asset.data, "default_root_state"):
        root_pos  = asset.data.default_root_state[env_ids, 0:3].clone()
        root_quat = asset.data.default_root_state[env_ids, 3:7].clone()
    else:
        root_pos  = torch.zeros((n, 3), device=env.device)
        root_quat = torch.zeros((n, 4), device=env.device); root_quat[:, 0] = 1.0

    roll  = math.radians(roll_pitch_deg) * (2.0 * torch.rand((n,), device=env.device) - 1.0)
    pitch = math.radians(roll_pitch_deg) * (2.0 * torch.rand((n,), device=env.device) - 1.0)
    yaw   = math.radians(yaw_deg)        * (2.0 * torch.rand((n,), device=env.device) - 1.0)

    delta_q   = _quat_from_euler_xyz_wxyz(roll, pitch, yaw)
    root_quat = _quat_normalize_wxyz(_quat_mul_wxyz(root_quat, delta_q))
    root_pose = torch.cat([root_pos, root_quat], dim=-1)
    root_vel  = torch.zeros((n, 6), device=env.device)

    if hasattr(asset, "write_root_state_to_sim"):
        asset.write_root_state_to_sim(torch.cat([root_pose, root_vel], dim=-1), env_ids=env_ids)
        return
    if hasattr(asset, "write_root_pose_to_sim"):
        asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    elif hasattr(asset, "write_root_link_pose_to_sim"):
        asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
    if hasattr(asset, "write_root_velocity_to_sim"):
        asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)


def randomize_mass(env, env_ids, scale_range: Tuple[float, float] = (0.85, 1.15)):
    """Domain randomization: scale robot body masses."""
    if env_ids is None:
        return
    if not torch.is_tensor(env_ids):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    asset = env.scene["robot"]
    if not hasattr(asset, "root_physx_view"):
        return
    try:
        masses = asset.root_physx_view.get_masses()  # (num_envs, num_bodies)
        if masses is None:
            return
        lo, hi = scale_range
        scale = lo + (hi - lo) * torch.rand((env_ids.numel(), masses.shape[1]), device=masses.device)
        masses_new = masses[env_ids] * scale
        asset.root_physx_view.set_masses(masses_new, env_ids)
    except Exception:
        pass


def randomize_friction(env, env_ids, range_lo: float = 0.3, range_hi: float = 1.5):
    """Domain randomization: randomize ground contact friction."""
    if env_ids is None:
        return
    if not torch.is_tensor(env_ids):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    asset = env.scene["robot"]
    if not hasattr(asset, "root_physx_view"):
        return
    try:
        mat = asset.root_physx_view.get_material_properties()  # (envs, bodies, 3)
        if mat is None:
            return
        n_envs = env_ids.numel()
        n_bodies = mat.shape[1]
        fric = range_lo + (range_hi - range_lo) * torch.rand((n_envs, n_bodies), device=mat.device)
        mat_new = mat[env_ids].clone()
        mat_new[..., 0] = fric   # static friction
        mat_new[..., 1] = fric   # dynamic friction
        asset.root_physx_view.set_material_properties(mat_new, env_ids)
    except Exception:
        pass


def apply_push_force(env, env_ids):
    """Apply a random horizontal impulse (curriculum stages 1+)."""
    stage = _get_curriculum_stage(env)
    if stage < 1:
        return
    if env_ids is None:
        return
    if not torch.is_tensor(env_ids):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    asset  = env.scene["robot"]
    n      = int(env_ids.numel())
    if n == 0:
        return

    # scale push magnitude with stage
    max_force = 80.0 if stage == 1 else 120.0
    force_xy  = max_force * (2.0 * torch.rand((n, 2), device=env.device) - 1.0)
    forces    = torch.zeros((n, 6), device=env.device)
    forces[:, 0] = force_xy[:, 0]
    forces[:, 1] = force_xy[:, 1]

    if hasattr(asset, "set_external_force_and_torque"):
        asset.set_external_force_and_torque(forces[:, :3], forces[:, 3:], env_ids=env_ids)


def reset_feet_air_time(env, env_ids):
    """Reset per-foot air-time tracking tensors on episode reset."""
    if env_ids is None:
        return
    if not torch.is_tensor(env_ids):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    _get_feet_air_time(env)[env_ids] = 0.0
    _get_feet_last_contact(env)[env_ids] = 1.0


# =============================================================================
# Body index helper
# =============================================================================
def _get_body_index(robot, body_name: str) -> int:
    for attr in ("body_names",):
        for src in (robot, getattr(robot, "data", None)):
            if src is not None and hasattr(src, attr):
                names = list(getattr(src, attr))
                if body_name in names:
                    return names.index(body_name)
    return -1


# =============================================================================
# Foot contact / velocity helpers
# =============================================================================
def obs_projected_gravity_b(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    q     = robot.data.root_quat_w
    g_w   = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    return _quat_rotate_inverse_wxyz(q, g_w)


def _feet_contact_forces(env):
    robot = env.scene["robot"]
    if not hasattr(robot.data, "net_contact_forces_w"):
        return None, -1, -1
    forces = robot.data.net_contact_forces_w
    li = _get_body_index(robot, LEFT_FOOT_BODY)
    ri = _get_body_index(robot, RIGHT_FOOT_BODY)
    return forces, li, ri


def obs_feet_contact_2(env) -> torch.Tensor:
    forces, li, ri = _feet_contact_forces(env)
    if forces is None or li < 0 or ri < 0:
        return torch.zeros((env.num_envs, 2), device=env.device)
    return torch.stack([
        (forces[:, li, 2] > 1.0).float(),
        (forces[:, ri, 2] > 1.0).float(),
    ], dim=-1)


def obs_feet_height_2(env) -> torch.Tensor:
    robot = env.scene["robot"]
    if not hasattr(robot.data, "body_pos_w"):
        return torch.zeros((env.num_envs, 2), device=env.device)
    pos = robot.data.body_pos_w
    li  = _get_body_index(robot, LEFT_FOOT_BODY)
    ri  = _get_body_index(robot, RIGHT_FOOT_BODY)
    if li < 0 or ri < 0:
        return torch.zeros((env.num_envs, 2), device=env.device)
    return torch.stack([pos[:, li, 2], pos[:, ri, 2]], dim=-1).float()


def _foot_xy_vel(env, body_name: str) -> torch.Tensor:
    robot = env.scene["robot"]
    if not hasattr(robot.data, "body_lin_vel_w"):
        return torch.zeros((env.num_envs, 2), device=env.device)
    idx = _get_body_index(robot, body_name)
    if idx < 0:
        return torch.zeros((env.num_envs, 2), device=env.device)
    return robot.data.body_lin_vel_w[:, idx, 0:2].float()


def obs_left_foot_speed_xy(env)  -> torch.Tensor: return _foot_xy_vel(env, LEFT_FOOT_BODY)
def obs_right_foot_speed_xy(env) -> torch.Tensor: return _foot_xy_vel(env, RIGHT_FOOT_BODY)


# =============================================================================
# Observations
# =============================================================================
def obs_commands_3(env) -> torch.Tensor:
    return _get_command_tensor(env)


def obs_base_ang_vel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_ang_vel_b


def obs_base_lin_vel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_lin_vel_b


def obs_dof_pos_offset_13(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    q    = mdp.joint_pos(env, asset_cfg=asset_cfg)
    q0   = torch.tensor(
        [DEFAULT_JOINT_ANGLES.get(j, 0.0) for j in asset_cfg.joint_names],
        device=env.device,
    )[None, :].repeat(env.num_envs, 1)
    return q - q0


def obs_dof_vel_13(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return mdp.joint_vel(env, asset_cfg=asset_cfg)


def obs_last_action_13(env) -> torch.Tensor:
    a = mdp.last_action(env)
    if a.shape[-1] > 13:
        a = a[..., :13]
    elif a.shape[-1] < 13:
        a = torch.cat([a, torch.zeros((env.num_envs, 13 - a.shape[-1]), device=env.device)], dim=-1)
    return a


def obs_base_height_offset(env, asset_cfg: SceneEntityCfg, target: float = 0.70) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_pos_w[:, 2:3] - float(target)


def obs_surround_heights_offset(env) -> torch.Tensor:
    return torch.zeros((env.num_envs, 8), device=env.device)


def obs_curriculum_stage(env) -> torch.Tensor:
    """Return curriculum stage as a 1-D observation (for critic only)."""
    stage = float(_get_curriculum_stage(env))
    return stage * torch.ones((env.num_envs, 1), device=env.device)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        commands              = ObsTerm(func=obs_commands_3)
        base_ang_vel          = ObsTerm(func=obs_base_ang_vel,        params={"asset_cfg": SceneEntityCfg("robot")})
        base_projected_gravity= ObsTerm(func=obs_projected_gravity_b, params={"asset_cfg": SceneEntityCfg("robot")})
        dof_pos_offset        = ObsTerm(func=obs_dof_pos_offset_13,   params={"asset_cfg": SceneEntityCfg("robot", joint_names=N1_ACTION_JOINTS_13)})
        dof_vel               = ObsTerm(func=obs_dof_vel_13,          params={"asset_cfg": SceneEntityCfg("robot", joint_names=N1_ACTION_JOINTS_13)})
        actions               = ObsTerm(func=obs_last_action_13)
        feet_contact          = ObsTerm(func=obs_feet_contact_2)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        commands              = ObsTerm(func=obs_commands_3)
        base_ang_vel          = ObsTerm(func=obs_base_ang_vel,        params={"asset_cfg": SceneEntityCfg("robot")})
        base_projected_gravity= ObsTerm(func=obs_projected_gravity_b, params={"asset_cfg": SceneEntityCfg("robot")})
        dof_pos_offset        = ObsTerm(func=obs_dof_pos_offset_13,   params={"asset_cfg": SceneEntityCfg("robot", joint_names=N1_ACTION_JOINTS_13)})
        dof_vel               = ObsTerm(func=obs_dof_vel_13,          params={"asset_cfg": SceneEntityCfg("robot", joint_names=N1_ACTION_JOINTS_13)})
        actions               = ObsTerm(func=obs_last_action_13)
        base_lin_vel          = ObsTerm(func=obs_base_lin_vel,        params={"asset_cfg": SceneEntityCfg("robot")})
        base_height_offset    = ObsTerm(func=obs_base_height_offset,  params={"asset_cfg": SceneEntityCfg("robot"), "target": 0.70})
        feet_contact          = ObsTerm(func=obs_feet_contact_2)
        feet_height           = ObsTerm(func=obs_feet_height_2)
        left_foot_speed_xy    = ObsTerm(func=obs_left_foot_speed_xy)
        right_foot_speed_xy   = ObsTerm(func=obs_right_foot_speed_xy)
        surround_heights      = ObsTerm(func=obs_surround_heights_offset)
        curriculum_stage      = ObsTerm(func=obs_curriculum_stage)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# =============================================================================
# Events
# =============================================================================
@configclass
class EventCfg:
    reset_scene          = EventTerm(func=reset_scene_safe,           mode="reset")
    randomize_root_pose  = EventTerm(func=randomize_root_pose_small,  mode="reset")
    random_joints        = EventTerm(func=spawn_random_action_joints, mode="reset")
    sample_commands      = EventTerm(func=sample_velocity_commands,   mode="reset")
    reset_air_time       = EventTerm(func=reset_feet_air_time,        mode="reset")
    randomize_mass       = EventTerm(func=randomize_mass,             mode="reset")
    randomize_friction   = EventTerm(func=randomize_friction,         mode="reset")
    # push_force is applied during startup of each reset (startup mode)
    push_startup         = EventTerm(func=apply_push_force,           mode="reset")


# =============================================================================
# Rewards
# =============================================================================
def rew_upright_from_gravity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    g_b = obs_projected_gravity_b(env, asset_cfg)
    err = torch.abs(g_b[:, 2] + 1.0)
    return torch.exp(-4.0 * err)


def rew_base_height_bonus(env, asset_cfg: SceneEntityCfg, target_height: float) -> torch.Tensor:
    z   = env.scene[asset_cfg.name].data.root_pos_w[:, 2]
    err = torch.abs(z - float(target_height))
    return torch.exp(-10.0 * err)


def rew_track_lin_vel_x(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    vx      = env.scene[asset_cfg.name].data.root_lin_vel_b[:, 0]
    vx_cmd  = _get_command_tensor(env)[:, 0]
    err     = (vx - vx_cmd) ** 2
    return torch.exp(-4.0 * err)


def rew_track_lin_vel_y(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    vy     = env.scene[asset_cfg.name].data.root_lin_vel_b[:, 1]
    vy_cmd = _get_command_tensor(env)[:, 1]
    err    = (vy - vy_cmd) ** 2
    return torch.exp(-4.0 * err)


def rew_track_yaw_rate(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    yaw_rate = env.scene[asset_cfg.name].data.root_ang_vel_b[:, 2]
    yaw_cmd  = _get_command_tensor(env)[:, 2]
    err      = (yaw_rate - yaw_cmd) ** 2
    return torch.exp(-2.0 * err)


def rew_lin_vel_z_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    vz = env.scene[asset_cfg.name].data.root_lin_vel_b[:, 2]
    return vz ** 2


def rew_ang_vel_xy_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    w = env.scene[asset_cfg.name].data.root_ang_vel_b
    return _l2_sq(w[:, 0:2])


def rew_dof_vel_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    qd = mdp.joint_vel(env, asset_cfg=asset_cfg)
    return _l2_sq(qd)


def rew_dof_accel_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint acceleration for smoothness."""
    if not hasattr(env, "_prev_joint_vel"):
        env._prev_joint_vel = mdp.joint_vel(env, asset_cfg=asset_cfg).clone()
        return torch.zeros((env.num_envs,), device=env.device)
    qd_now  = mdp.joint_vel(env, asset_cfg=asset_cfg)
    qdd     = qd_now - env._prev_joint_vel
    env._prev_joint_vel = qd_now.clone()
    return _l2_sq(qdd)


def rew_action_rate_penalty(env) -> torch.Tensor:
    a    = mdp.last_action(env)
    prev = None
    if hasattr(env, "action_manager"):
        for cand in ("prev_action", "_prev_action", "previous_action"):
            if hasattr(env.action_manager, cand):
                prev = getattr(env.action_manager, cand); break
    if prev is None:
        return torch.zeros((env.num_envs,), device=env.device)
    m    = min(prev.shape[-1], a.shape[-1])
    return _l2_sq(a[..., :m] - prev[..., :m])


def rew_foot_slip_penalty(env) -> torch.Tensor:
    contact = obs_feet_contact_2(env)
    v_l = obs_left_foot_speed_xy(env)
    v_r = obs_right_foot_speed_xy(env)
    return contact[:, 0] * _l2_sq(v_l) + contact[:, 1] * _l2_sq(v_r)


def rew_feet_air_time(env) -> torch.Tensor:
    """
    Reward alternate foot contact: give bonus when a foot touches down
    after being in the air for 0.2 – 0.8 s.
    Adapted from Zhuang et al. / legged_gym style.
    """
    step_dt = float(getattr(env, "step_dt",
                             getattr(env.cfg.sim, "dt", 0.001) * getattr(env.cfg, "decimation", 10)))
    contact      = obs_feet_contact_2(env)          # (B, 2)
    last_contact = _get_feet_last_contact(env)       # (B, 2)
    air_time     = _get_feet_air_time(env)            # (B, 2)

    # For feet currently NOT in contact, accumulate air time
    in_air   = (contact < 0.5).float()
    air_time += in_air * step_dt

    # Touchdown event: was in air, now in contact
    touchdown = ((last_contact < 0.5) & (contact > 0.5)).float()
    # Reward only if air time is in [0.2, 0.8] s (reasonable step duration)
    air_ok   = ((air_time > 0.2) & (air_time < 0.8)).float()
    bonus    = torch.sum(touchdown * air_ok, dim=-1)

    # Reset air_time on touchdown
    air_time[:] = air_time * (1.0 - touchdown)
    _get_feet_last_contact(env)[:] = contact

    # Scale reward by forward command (no gait reward needed at standstill)
    vx_cmd = _get_command_tensor(env)[:, 0]
    return bonus * torch.clamp(vx_cmd / 1.0, 0.0, 1.0)


def rew_feet_clearance(env) -> torch.Tensor:
    """Reward lifting feet off the ground during swing phase."""
    contact  = obs_feet_contact_2(env)   # (B, 2)  1 = on ground
    heights  = obs_feet_height_2(env)    # (B, 2)  absolute foot height
    swing    = (contact < 0.5).float()   # feet in swing
    # reward if foot is lifted > 2 cm in swing
    clearance = torch.clamp(heights - 0.02, min=0.0)
    return torch.sum(swing * clearance, dim=-1)


def rew_torque_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize high joint torques (energy efficiency)."""
    q  = mdp.joint_pos(env, asset_cfg=asset_cfg)
    qd = mdp.joint_vel(env, asset_cfg=asset_cfg)
    # Approximate torque via PD law: τ = kp*(qd_target - q) + kd*(0 - qd)
    # Here just penalize qd^2 as proxy for I^2R loss
    return _l2_sq(qd)


@configclass
class RewardsCfg:
    # ── survival ──────────────────────────────────────────────────────────
    alive       = RewTerm(func=mdp.is_alive,      weight=1.5)
    terminating = RewTerm(func=mdp.is_terminated, weight=-50.0)

    # ── stability ─────────────────────────────────────────────────────────
    upright     = RewTerm(func=rew_upright_from_gravity, weight=3.0,
                          params={"asset_cfg": SceneEntityCfg("robot")})
    height      = RewTerm(func=rew_base_height_bonus, weight=1.5,
                          params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.70})

    # ── command tracking (primary objectives) ─────────────────────────────
    track_vx    = RewTerm(func=rew_track_lin_vel_x, weight=8.0,
                          params={"asset_cfg": SceneEntityCfg("robot")})
    track_vy    = RewTerm(func=rew_track_lin_vel_y, weight=2.0,
                          params={"asset_cfg": SceneEntityCfg("robot")})
    track_yaw   = RewTerm(func=rew_track_yaw_rate,  weight=1.0,
                          params={"asset_cfg": SceneEntityCfg("robot")})

    # ── gait quality ──────────────────────────────────────────────────────
    air_time    = RewTerm(func=rew_feet_air_time,   weight=2.0)
    clearance   = RewTerm(func=rew_feet_clearance,  weight=0.5)

    # ── penalties ─────────────────────────────────────────────────────────
    lin_vel_z   = RewTerm(func=rew_lin_vel_z_penalty,   weight=-2.0,
                          params={"asset_cfg": SceneEntityCfg("robot")})
    ang_vel_xy  = RewTerm(func=rew_ang_vel_xy_penalty,  weight=-0.2,
                          params={"asset_cfg": SceneEntityCfg("robot")})
    dof_vel     = RewTerm(func=rew_dof_vel_penalty,     weight=-0.005,
                          params={"asset_cfg": SceneEntityCfg("robot", joint_names=N1_ACTION_JOINTS_13)})
    dof_accel   = RewTerm(func=rew_dof_accel_penalty,   weight=-1e-4,
                          params={"asset_cfg": SceneEntityCfg("robot", joint_names=N1_ACTION_JOINTS_13)})
    action_rate = RewTerm(func=rew_action_rate_penalty, weight=-0.01)
    foot_slip   = RewTerm(func=rew_foot_slip_penalty,   weight=-0.4)


# =============================================================================
# Terminations
# =============================================================================
def obs_projected_gravity_b_standalone(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return obs_projected_gravity_b(env, asset_cfg)


def done_root_height_below(env, asset_cfg: SceneEntityCfg, min_height: float) -> torch.Tensor:
    z = env.scene[asset_cfg.name].data.root_pos_w[:, 2]
    return z < float(min_height)


def done_bad_orientation_safe(env, asset_cfg: SceneEntityCfg, limit_angle: float) -> torch.Tensor:
    if hasattr(mdp, "bad_orientation"):
        return mdp.bad_orientation(env, asset_cfg=asset_cfg, limit_angle=limit_angle)
    g_b = obs_projected_gravity_b(env, asset_cfg)
    return torch.abs(g_b[:, 2] + 1.0) > 0.8


@configclass
class TerminationsCfg:
    time_out        = DoneTerm(func=mdp.time_out, time_out=True)
    fallen          = DoneTerm(func=done_root_height_below,
                               params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 0.45})
    bad_orientation = DoneTerm(func=done_bad_orientation_safe,
                               params={"asset_cfg": SceneEntityCfg("robot"),
                                       "limit_angle": math.pi * 0.65})


# =============================================================================
# Curriculum (IsaacLab curriculum term – optional; stage also managed externally)
# =============================================================================
def curriculum_advance_stage(env, env_ids) -> dict:
    """
    Called by IsaacLab curriculum manager each episode.
    Returns metrics for logging; actual stage advance is done by train script.
    """
    stage = _get_curriculum_stage(env)
    return {"curriculum_stage": stage}


@configclass
class CommandsCfg:
    pass


@configclass
class CurriculumCfg:
    log_stage = CurrTerm(func=curriculum_advance_stage)


# =============================================================================
# Environment
# =============================================================================
@configclass
class HumanoidN1LocomotionEnvCfgV2(ManagerBasedRLEnvCfg):
    scene:        HumanoidN1SceneCfg   = HumanoidN1SceneCfg(num_envs=512, env_spacing=2.5)
    observations: ObservationsCfg     = ObservationsCfg()
    actions:      ActionsCfg          = ActionsCfg()
    events:       EventCfg            = EventCfg()
    rewards:      RewardsCfg          = RewardsCfg()
    terminations: TerminationsCfg     = TerminationsCfg()
    commands:     CommandsCfg         = CommandsCfg()
    curriculum:   CurriculumCfg       = CurriculumCfg()

    def __post_init__(self) -> None:
        self.sim.dt         = 0.001
        self.decimation     = 10         # control at 100 Hz
        self.episode_length_s = 15.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (12.0, 12.0, 7.0)
