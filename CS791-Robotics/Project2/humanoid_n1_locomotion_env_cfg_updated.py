# humanoid_n1_locomotion_env_cfg.py
# Minimal locomotion task for N1 based on stand task

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
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


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


# =============================================================================
# Foot bodies
# =============================================================================
LEFT_FOOT_BODY = "left_foot_pitch_link"
RIGHT_FOOT_BODY = "right_foot_pitch_link"


# =============================================================================
# Joint limits for 13 actions
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

assert ACTIONS_MAX_13.shape == (13,)
assert ACTIONS_MIN_13.shape == (13,)


# =============================================================================
# Default joint angles (standing pose)
# =============================================================================
DEFAULT_JOINT_ANGLES = {
    "left_hip_pitch_joint": float(-np.deg2rad(14.0)),
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_pitch_joint": float(+np.deg2rad(29.5)),
    "left_ankle_roll_joint": 0.0,
    "left_ankle_pitch_joint": float(-np.deg2rad(13.7)),

    "right_hip_pitch_joint": float(-np.deg2rad(14.0)),
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
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
ACTION_SCALE_BY_JOINT = {}
for j, jmax, jmin in zip(N1_ACTION_JOINTS_13, ACTIONS_MAX_13, ACTIONS_MIN_13):
    q0 = float(DEFAULT_JOINT_ANGLES.get(j, 0.0))
    s_pos = float(jmax) - q0
    s_neg = q0 - float(jmin)
    scale = max(1e-6, min(s_pos, s_neg))
    ACTION_OFFSET_BY_JOINT[j] = q0
    ACTION_SCALE_BY_JOINT[j] = float(scale)


# =============================================================================
# Torque limits (23)
# =============================================================================
TAU_LIMIT_23 = np.array([
    95, 54, 54, 95, 30, 30,
    95, 54, 54, 95, 30, 30,
    54,
    54, 30, 30, 30, 30,
    54, 30, 30, 30, 30,
], dtype=np.float32)
TAU_LIMIT_BY_JOINT = {j: float(tau) for j, tau in zip(N1_JOINTS_23, TAU_LIMIT_23)}


# =============================================================================
# PD gains
# =============================================================================
def kp_kd_for_joint(j: str) -> Tuple[float, float]:
    if "hip_pitch" in j:
        return 180.0, 10.0
    if "hip_roll" in j:
        return 120.0, 10.0
    if "hip_yaw" in j:
        return 90.0, 8.0
    if "knee" in j:
        return 120.0, 8.0
    if "ankle" in j:
        return 45.0, 2.5
    if "waist" in j:
        return 90.0, 8.0
    if "shoulder_pitch" in j:
        return 90.0, 8.0
    return 45.0, 2.5


ACTUATORS = {
    f"act_{j}": ImplicitActuatorCfg(
        joint_names_expr=[j],
        stiffness=float(kp_kd_for_joint(j)[0]),
        damping=float(kp_kd_for_joint(j)[1]),
        effort_limit_sim=float(TAU_LIMIT_BY_JOINT.get(j, 300.0)),
        velocity_limit_sim=50.0,
    )
    for j in N1_JOINTS_23
}


# =============================================================================
# Math helpers
# =============================================================================
@torch.jit.script
def _quat_rotate_inverse_wxyz(q_wxyz: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
    w = q_wxyz[:, 0:1]
    xyz = q_wxyz[:, 1:4]
    t = 2.0 * torch.cross(xyz, v_w, dim=1)
    return v_w + w * t + torch.cross(xyz, t, dim=1)


def _l2_sq(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * x, dim=-1)


def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication for tensors in (w, x, y, z) format."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _quat_normalize_wxyz(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-8)


def _quat_from_euler_xyz_wxyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    return torch.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dim=-1,
    )


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
        joint_vel={j: 0.0 for j in DEFAULT_JOINT_ANGLES.keys()},
    ),
    actuators=ACTUATORS,
)


@configclass
class HumanoidN1SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
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
# Command utilities
# =============================================================================
def _get_command_tensor(env) -> torch.Tensor:
    if not hasattr(env, "_n1_commands"):
        env._n1_commands = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    return env._n1_commands


def sample_velocity_commands(env, env_ids):
    """Sample [vx_cmd, vy_cmd, yaw_rate_cmd]. Minimal version uses vx only."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif not torch.is_tensor(env_ids):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    cmds = _get_command_tensor(env)

    # forward velocity command in [0.0, 0.5] m/s
    vx = 0.5 * torch.rand((env_ids.numel(),), device=env.device)

    # keep these zero for a minimal homework locomotion task
    vy = torch.zeros_like(vx)
    yaw = torch.zeros_like(vx)

    cmds[env_ids, 0] = vx
    cmds[env_ids, 1] = vy
    cmds[env_ids, 2] = yaw


# =============================================================================
# Reset events
# =============================================================================
def _get_articulation_joint_names(asset) -> List[str]:
    for attr in ("joint_names", "dof_names"):
        if hasattr(asset, attr):
            return list(getattr(asset, attr))
        if hasattr(asset, "data") and hasattr(asset.data, attr):
            return list(getattr(asset.data, attr))
    raise AttributeError("Could not find joint/dof names on articulation.")


def spawn_random_action_joints(env, env_ids):
    if not torch.is_tensor(env_ids):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    asset = env.scene["robot"]
    all_joint_names = _get_articulation_joint_names(asset)

    num_envs = int(env_ids.numel())
    dof_count = len(all_joint_names)

    qpos = torch.zeros((num_envs, dof_count), device=env.device, dtype=torch.float32)
    qvel = torch.zeros((num_envs, dof_count), device=env.device, dtype=torch.float32)

    for i, jname in enumerate(all_joint_names):
        qpos[:, i] = float(DEFAULT_JOINT_ANGLES.get(jname, 0.0))

    for i, jname in enumerate(all_joint_names):
        if jname in N1_ACTION_JOINTS_13:
            s = float(ACTION_SCALE_BY_JOINT[jname])
            delta = (0.35 * s) * (2.0 * torch.rand((num_envs,), device=env.device) - 1.0)
            qpos[:, i] = qpos[:, i] + delta

    if hasattr(asset, "write_joint_positions_to_sim"):
        asset.write_joint_positions_to_sim(qpos, env_ids=env_ids)
    elif hasattr(asset, "write_joint_state_to_sim"):
        asset.write_joint_state_to_sim(qpos, qvel, env_ids=env_ids)

    if hasattr(asset, "write_joint_velocities_to_sim"):
        asset.write_joint_velocities_to_sim(qvel, env_ids=env_ids)


def reset_scene_safe(env, env_ids=None):
    if hasattr(mdp, "reset_scene_to_default"):
        return mdp.reset_scene_to_default(env, env_ids)
    if hasattr(mdp, "reset_scene"):
        return mdp.reset_scene(env, env_ids)
    return None


def randomize_root_pose_small(env, env_ids, roll_pitch_deg: float = 3.0, yaw_deg: float = 10.0):
    """Apply small base orientation perturbations at reset.

    This adds the rubric-requested base yaw plus small roll/pitch randomization.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif not torch.is_tensor(env_ids):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    asset = env.scene["robot"]
    num_envs = int(env_ids.numel())
    if num_envs == 0:
        return

    if hasattr(asset.data, "root_pos_w"):
        root_pos = asset.data.root_pos_w[env_ids].clone()
    elif hasattr(asset.data, "default_root_state"):
        root_pos = asset.data.default_root_state[env_ids, 0:3].clone()
    else:
        root_pos = torch.zeros((num_envs, 3), device=env.device, dtype=torch.float32)

    if hasattr(asset.data, "root_quat_w"):
        root_quat = asset.data.root_quat_w[env_ids].clone()
    elif hasattr(asset.data, "default_root_state"):
        root_quat = asset.data.default_root_state[env_ids, 3:7].clone()
    else:
        root_quat = torch.zeros((num_envs, 4), device=env.device, dtype=torch.float32)
        root_quat[:, 0] = 1.0

    roll = math.radians(roll_pitch_deg) * (2.0 * torch.rand((num_envs,), device=env.device) - 1.0)
    pitch = math.radians(roll_pitch_deg) * (2.0 * torch.rand((num_envs,), device=env.device) - 1.0)
    yaw = math.radians(yaw_deg) * (2.0 * torch.rand((num_envs,), device=env.device) - 1.0)

    delta_quat = _quat_from_euler_xyz_wxyz(roll, pitch, yaw)
    root_quat = _quat_normalize_wxyz(_quat_mul_wxyz(root_quat, delta_quat))

    root_pose = torch.cat([root_pos, root_quat], dim=-1)
    root_vel = torch.zeros((num_envs, 6), device=env.device, dtype=torch.float32)

    if hasattr(asset, "write_root_state_to_sim"):
        root_state = torch.cat([root_pose, root_vel], dim=-1)
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)
        return

    if hasattr(asset, "write_root_pose_to_sim"):
        asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    elif hasattr(asset, "write_root_link_pose_to_sim"):
        asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

    if hasattr(asset, "write_root_velocity_to_sim"):
        asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    elif hasattr(asset, "write_root_link_velocity_to_sim"):
        asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)


# =============================================================================
# Observations
# =============================================================================
def obs_commands_3(env) -> torch.Tensor:
    return _get_command_tensor(env)


def obs_base_ang_vel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return robot.data.root_ang_vel_b


def obs_base_lin_vel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return robot.data.root_lin_vel_b


def obs_projected_gravity_b(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    q = robot.data.root_quat_w
    g_w = torch.tensor([0.0, 0.0, -1.0], device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
    return _quat_rotate_inverse_wxyz(q, g_w)


def obs_dof_pos_offset_13(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    q = mdp.joint_pos(env, asset_cfg=asset_cfg)
    q0_vals = [DEFAULT_JOINT_ANGLES.get(j, 0.0) for j in asset_cfg.joint_names]
    q0 = torch.tensor(q0_vals, device=env.device, dtype=torch.float32)[None, :].repeat(env.num_envs, 1)
    return q - q0


def obs_dof_vel_13(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return mdp.joint_vel(env, asset_cfg=asset_cfg)


def obs_last_action_13(env) -> torch.Tensor:
    a = mdp.last_action(env)
    if a.shape[-1] > 13:
        a = a[..., :13]
    elif a.shape[-1] < 13:
        pad = torch.zeros((env.num_envs, 13 - a.shape[-1]), device=env.device, dtype=torch.float32)
        a = torch.cat([a, pad], dim=-1)
    return a


def obs_base_height_offset(env, asset_cfg: SceneEntityCfg, target: float = 0.70) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    z = robot.data.root_pos_w[:, 2:3]
    return z - float(target)


# =============================================================================
# Foot measurements
# =============================================================================
def _get_body_index(robot, body_name: str) -> int:
    if hasattr(robot.data, "body_names"):
        names = list(robot.data.body_names)
        if body_name in names:
            return names.index(body_name)
    if hasattr(robot, "body_names"):
        names = list(robot.body_names)
        if body_name in names:
            return names.index(body_name)
    return -1


def obs_feet_contact_2(env) -> torch.Tensor:
    robot = env.scene["robot"]
    if not hasattr(robot.data, "net_contact_forces_w"):
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)

    forces = robot.data.net_contact_forces_w
    li = _get_body_index(robot, LEFT_FOOT_BODY)
    ri = _get_body_index(robot, RIGHT_FOOT_BODY)
    if li < 0 or ri < 0:
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)

    fz_l = forces[:, li, 2]
    fz_r = forces[:, ri, 2]
    contact_l = (fz_l > 1.0).to(torch.float32)
    contact_r = (fz_r > 1.0).to(torch.float32)
    return torch.stack([contact_l, contact_r], dim=-1)


def obs_feet_height_2(env) -> torch.Tensor:
    robot = env.scene["robot"]
    if not hasattr(robot.data, "body_pos_w"):
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)

    pos = robot.data.body_pos_w
    li = _get_body_index(robot, LEFT_FOOT_BODY)
    ri = _get_body_index(robot, RIGHT_FOOT_BODY)
    if li < 0 or ri < 0:
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)

    zl = pos[:, li, 2]
    zr = pos[:, ri, 2]
    return torch.stack([zl, zr], dim=-1).to(torch.float32)


def obs_left_foot_speed_xy(env) -> torch.Tensor:
    robot = env.scene["robot"]
    if not hasattr(robot.data, "body_lin_vel_w"):
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    vel = robot.data.body_lin_vel_w
    li = _get_body_index(robot, LEFT_FOOT_BODY)
    if li < 0:
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    return vel[:, li, 0:2].to(torch.float32)


def obs_right_foot_speed_xy(env) -> torch.Tensor:
    robot = env.scene["robot"]
    if not hasattr(robot.data, "body_lin_vel_w"):
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    vel = robot.data.body_lin_vel_w
    ri = _get_body_index(robot, RIGHT_FOOT_BODY)
    if ri < 0:
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    return vel[:, ri, 0:2].to(torch.float32)


def obs_surround_heights_offset(env) -> torch.Tensor:
    return torch.zeros((env.num_envs, 8), device=env.device, dtype=torch.float32)


# =============================================================================
# Observation groups
# =============================================================================
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        commands = ObsTerm(func=obs_commands_3)
        base_ang_vel = ObsTerm(func=obs_base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_projected_gravity = ObsTerm(func=obs_projected_gravity_b, params={"asset_cfg": SceneEntityCfg("robot")})
        dof_pos_offset = ObsTerm(func=obs_dof_pos_offset_13, params={"asset_cfg": SceneEntityCfg("robot", joint_names=N1_ACTION_JOINTS_13)})
        dof_vel = ObsTerm(func=obs_dof_vel_13, params={"asset_cfg": SceneEntityCfg("robot", joint_names=N1_ACTION_JOINTS_13)})
        actions = ObsTerm(func=obs_last_action_13)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        commands = ObsTerm(func=obs_commands_3)
        base_ang_vel = ObsTerm(func=obs_base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_projected_gravity = ObsTerm(func=obs_projected_gravity_b, params={"asset_cfg": SceneEntityCfg("robot")})
        dof_pos_offset = ObsTerm(func=obs_dof_pos_offset_13, params={"asset_cfg": SceneEntityCfg("robot", joint_names=N1_ACTION_JOINTS_13)})
        dof_vel = ObsTerm(func=obs_dof_vel_13, params={"asset_cfg": SceneEntityCfg("robot", joint_names=N1_ACTION_JOINTS_13)})
        actions = ObsTerm(func=obs_last_action_13)
        base_lin_vel = ObsTerm(func=obs_base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_heights_offset = ObsTerm(func=obs_base_height_offset, params={"asset_cfg": SceneEntityCfg("robot"), "target": 0.70})
        feet_contact = ObsTerm(func=obs_feet_contact_2)
        feet_height = ObsTerm(func=obs_feet_height_2)
        left_foot_speed_xy = ObsTerm(func=obs_left_foot_speed_xy)
        right_foot_speed_xy = ObsTerm(func=obs_right_foot_speed_xy)
        surround_heights_offset = ObsTerm(func=obs_surround_heights_offset)

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
    reset_scene = EventTerm(func=reset_scene_safe, mode="reset")
    randomize_root_pose = EventTerm(func=randomize_root_pose_small, mode="reset")
    random_action_joints = EventTerm(func=spawn_random_action_joints, mode="reset")
    sample_commands = EventTerm(func=sample_velocity_commands, mode="reset")


# =============================================================================
# Rewards
# =============================================================================
def rew_upright_from_gravity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    g_b = obs_projected_gravity_b(env, asset_cfg)
    err = torch.abs(g_b[:, 2] + 1.0)
    return torch.exp(-4.0 * err)


def rew_base_height_bonus(env, asset_cfg: SceneEntityCfg, target_height: float) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    z = robot.data.root_pos_w[:, 2]
    err = torch.abs(z - float(target_height))
    return torch.exp(-12.0 * err)


def rew_pose_similarity_13(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    q = mdp.joint_pos(env, asset_cfg=asset_cfg)
    joint_names = list(getattr(asset_cfg, "joint_names", []) or [])
    q0_vals = [DEFAULT_JOINT_ANGLES.get(j, 0.0) for j in joint_names]
    q0 = torch.tensor(q0_vals, device=env.device, dtype=torch.float32)[None, :].repeat(env.num_envs, 1)

    # In some debug calls Isaac Lab may still return the full articulation state.
    # Safely align dimensions to the controlled-joint reference vector.
    if q.ndim != 2:
        q = q.view(q.shape[0], -1)
    m = min(q.shape[1], q0.shape[1])
    if m == 0:
        return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
    q = q[:, :m]
    q0 = q0[:, :m]

    err = torch.mean(torch.abs(q - q0), dim=1)
    return torch.exp(-1.0 * err)


def rew_track_lin_vel_x(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    vx = robot.data.root_lin_vel_b[:, 0]
    vx_cmd = _get_command_tensor(env)[:, 0]
    err = (vx - vx_cmd) ** 2
    return torch.exp(-4.0 * err)


def rew_track_yaw_rate(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    yaw_rate = robot.data.root_ang_vel_b[:, 2]
    yaw_cmd = _get_command_tensor(env)[:, 2]
    err = (yaw_rate - yaw_cmd) ** 2
    return torch.exp(-2.0 * err)


def rew_lin_vel_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    v = obs_base_lin_vel(env, asset_cfg)
    # penalize lateral / vertical motion more than forward motion
    vy_vz = v[:, 1:3]
    return _l2_sq(vy_vz)


def rew_ang_vel_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    w = obs_base_ang_vel(env, asset_cfg)
    # penalize roll/pitch rate more than yaw rate for locomotion
    return _l2_sq(w[:, 0:2])


def rew_dof_vel_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    qd = mdp.joint_vel(env, asset_cfg=asset_cfg)
    return _l2_sq(qd)


def rew_action_rate_penalty(env) -> torch.Tensor:
    a = mdp.last_action(env)
    prev = None
    if hasattr(env, "action_manager"):
        am = env.action_manager
        for cand in ("prev_action", "_prev_action", "previous_action"):
            if hasattr(am, cand):
                prev = getattr(am, cand)
                break
    if prev is None:
        return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
    m = min(prev.shape[-1], a.shape[-1])
    diff = a[..., :m] - prev[..., :m]
    return _l2_sq(diff)


def rew_foot_slip_penalty(env) -> torch.Tensor:
    contacts = obs_feet_contact_2(env)
    v_l = obs_left_foot_speed_xy(env)
    v_r = obs_right_foot_speed_xy(env)

    slip_l = contacts[:, 0] * torch.sum(v_l * v_l, dim=1)
    slip_r = contacts[:, 1] * torch.sum(v_r * v_r, dim=1)
    return slip_l + slip_r


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=2.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-50.0)

    upright = RewTerm(func=rew_upright_from_gravity, weight=4.0, params={"asset_cfg": SceneEntityCfg("robot")})
    height = RewTerm(func=rew_base_height_bonus, weight=2.5, params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.70})

    # keep this small so motion is allowed
    pose = RewTerm(func=rew_pose_similarity_13, weight=0.75, params={"asset_cfg": SceneEntityCfg("robot", joint_names=N1_ACTION_JOINTS_13)})

    # locomotion-specific rewards
    track_lin_vel_x = RewTerm(func=rew_track_lin_vel_x, weight=6.0, params={"asset_cfg": SceneEntityCfg("robot")})
    track_yaw_rate = RewTerm(func=rew_track_yaw_rate, weight=0.5, params={"asset_cfg": SceneEntityCfg("robot")})

    # penalties
    lin_vel = RewTerm(func=rew_lin_vel_penalty, weight=-0.4, params={"asset_cfg": SceneEntityCfg("robot")})
    ang_vel = RewTerm(func=rew_ang_vel_penalty, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot")})
    dof_vel = RewTerm(func=rew_dof_vel_penalty, weight=-0.01, params={"asset_cfg": SceneEntityCfg("robot", joint_names=N1_ACTION_JOINTS_13)})
    action_rate = RewTerm(func=rew_action_rate_penalty, weight=-0.01)
    foot_slip = RewTerm(func=rew_foot_slip_penalty, weight=-0.5)


# =============================================================================
# Terminations
# =============================================================================
def done_root_height_below(env, asset_cfg: SceneEntityCfg, min_height: float) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    z = robot.data.root_pos_w[:, 2]
    return z < float(min_height)


def done_bad_orientation_safe(env, asset_cfg: SceneEntityCfg, limit_angle: float) -> torch.Tensor:
    if hasattr(mdp, "bad_orientation"):
        return mdp.bad_orientation(env, asset_cfg=asset_cfg, limit_angle=limit_angle)
    g_b = obs_projected_gravity_b(env, asset_cfg)
    err = torch.abs(g_b[:, 2] + 1.0)
    return err > 0.8


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fallen = DoneTerm(func=done_root_height_below, params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 0.50})
    bad_orientation = DoneTerm(func=done_bad_orientation_safe, params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": math.pi * 0.65})


@configclass
class CommandsCfg:
    """Placeholder only. Actual command tensor is managed in reset events."""
    pass


@configclass
class CurriculumCfg:
    pass


# =============================================================================
# Environment
# =============================================================================
@configclass
class HumanoidN1LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    scene: HumanoidN1SceneCfg = HumanoidN1SceneCfg(num_envs=256, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.sim.dt = 0.001
        self.decimation = 10
        self.episode_length_s = 12.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (12.0, 12.0, 7.0)
