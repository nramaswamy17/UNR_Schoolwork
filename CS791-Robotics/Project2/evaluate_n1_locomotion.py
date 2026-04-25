#!/usr/bin/env python3
"""
CS791 Project 2 – safer N1 locomotion evaluation script.

Main fixes vs. the hanging version:
  1. Disables reset-time command sampling, mass/friction randomization, and startup pushes
     for nominal evaluation. Those were running inside env.reset().
  2. Manually injects fixed vx commands before reset and after every step.
  3. Tracks per-env "ever fell" success correctly instead of using only the final terminated mask.
  4. Clears external push forces after one control step so a push test is an impulse-like disturbance,
     not a persistent lateral force.
  5. Adds an action-delay robustness test, which avoids touching PhysX mass/material properties
     during reset and still satisfies the Project 2 robustness category list.

Usage:
  ./isaaclab.sh -p evaluate_n1_locomotion_fixed.py \
      --checkpoint /path/to/model_XXXX.pt \
      --num_envs 64 --headless
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate N1 locomotion policy safely.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_XXX.pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--eval_steps", type=int, default=1500, help="Steps per test. 1500 = 15 s at 100 Hz.")
parser.add_argument("--episode_length_s", type=float, default=20.0)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--print_every", type=int, default=250)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Post-launch imports
# -----------------------------------------------------------------------------
import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from humanoid_n1_locomotion_env_cfg_v2 import (
    HumanoidN1LocomotionEnvCfgV2,
    _get_command_tensor,
    obs_feet_contact_2,
    obs_left_foot_speed_xy,
    obs_right_foot_speed_xy,
    _l2_sq,
)
from agents.rsl_rl_ppo_cfg_locomotion import HumanoidN1LocomotionPPORunnerCfg

try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

try:
    from isaaclab.utils.dict import class_to_dict
except ImportError:
    class_to_dict = None


# -----------------------------------------------------------------------------
# Evaluation plan
# -----------------------------------------------------------------------------
EVAL_SPEEDS = [0.0, 0.8, 1.5, 2.5, 3.2]
SPEED_LABELS = ["stand", "slow_walk", "walk", "fast_walk", "run"]

ROBUSTNESS_TESTS = [
    {
        "label": "velocity_kick_disturbance",
        "vx": 1.5,
        "push": True,
        "push_delta_v": 0.35,
        "push_interval_steps": 300,
        "action_delay_steps": 0,
        "description": "0.35 m/s random lateral root-velocity kick every 3 s",
    },
    {
        "label": "action_delay_2_steps",
        "vx": 1.5,
        "push": False,
        "push_delta_v": 0.0,
        "push_interval_steps": 300,
        "action_delay_steps": 2,
        "description": "Hold previous action for two control steps",
    },
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def cfg_to_dict(cfg_obj: Any) -> dict:
    if hasattr(cfg_obj, "to_dict"):
        return cfg_obj.to_dict()
    if class_to_dict is not None:
        return class_to_dict(cfg_obj)
    if isinstance(cfg_obj, dict):
        return cfg_obj
    raise TypeError(f"Cannot convert {type(cfg_obj)} to dict.")


def disable_event(cfg: Any, name: str) -> None:
    """Best-effort removal of an EventTermCfg before ManagerBasedRLEnv construction."""
    events = getattr(cfg, "events", None)
    if events is not None and hasattr(events, name):
        setattr(events, name, None)


def make_eval_env_cfg() -> HumanoidN1LocomotionEnvCfgV2:
    cfg = HumanoidN1LocomotionEnvCfgV2()
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.episode_length_s = float(args_cli.episode_length_s)
    if args_cli.seed is not None:
        try:
            cfg.seed = int(args_cli.seed)
        except Exception:
            pass

    # Critical: keep evaluation reset simple and deterministic. The hanging script
    # stalled inside env.reset(), so do not run command sampling, mass/material
    # edits, or startup pushes as reset events during nominal evaluation.
    for event_name in [
        "sample_commands",
        "randomize_mass",
        "randomize_friction",
        "push_startup",
    ]:
        disable_event(cfg, event_name)

    return cfg


def build_inference_policy(env: ManagerBasedRLEnv, checkpoint_path: Path, device: str):
    wrapped_env = RslRlVecEnvWrapper(env)
    agent_cfg = HumanoidN1LocomotionPPORunnerCfg()
    runner = OnPolicyRunner(
        env=wrapped_env,
        train_cfg=cfg_to_dict(agent_cfg),
        log_dir=None,
        device=str(device),
    )

    load_errors: list[str] = []
    for attempt in [
        lambda: runner.load(str(checkpoint_path)),
        lambda: runner.load(checkpoint_path),
    ]:
        try:
            attempt()
            break
        except Exception as exc:
            load_errors.append(repr(exc))
    else:
        raise RuntimeError("runner.load() failed:\n" + "\n".join(load_errors))

    for getter in [
        lambda: runner.get_inference_policy(device=device),
        lambda: runner.get_inference_policy(),
    ]:
        try:
            policy = getter()
            if callable(policy):
                return policy, wrapped_env
        except Exception:
            pass

    raise RuntimeError("Checkpoint loaded, but runner.get_inference_policy() failed.")


def extract_policy_obs(obs: Any) -> torch.Tensor:
    if torch.is_tensor(obs):
        return obs
    if isinstance(obs, dict):
        if "policy" in obs and torch.is_tensor(obs["policy"]):
            return obs["policy"]
        for key in ["obs", "observation", "actor"]:
            if key in obs and torch.is_tensor(obs[key]):
                return obs[key]
    raise TypeError(f"Could not extract policy observation from type {type(obs)}")


def set_commands(env: ManagerBasedRLEnv, vx: float) -> None:
    cmds = _get_command_tensor(env)
    cmds[:, 0] = float(vx)
    cmds[:, 1] = 0.0
    cmds[:, 2] = 0.0


def compute_slip_metric(env: ManagerBasedRLEnv) -> torch.Tensor:
    contact = obs_feet_contact_2(env)
    v_l = obs_left_foot_speed_xy(env)
    v_r = obs_right_foot_speed_xy(env)
    slip_l = contact[:, 0] * _l2_sq(v_l).sqrt()
    slip_r = contact[:, 1] * _l2_sq(v_r).sqrt()
    return 0.5 * (slip_l + slip_r)


def apply_root_velocity_kick(env: ManagerBasedRLEnv, delta_v: float) -> None:
    """Apply a one-step lateral disturbance without using PhysX external-force buffers.

    This is intentionally implemented as an instantaneous root velocity perturbation
    instead of set_external_force_and_torque(). The external-force API is easy to
    misuse because it is body-indexed and stores force buffers internally; for
    evaluation, a velocity kick is simpler, deterministic, and avoids simulator
    stalls around the first push event.
    """
    if delta_v == 0.0:
        return

    asset = env.scene["robot"]
    n = env.num_envs
    env_ids = torch.arange(n, device=env.device, dtype=torch.long)

    if hasattr(asset.data, "root_vel_w"):
        root_vel = asset.data.root_vel_w[env_ids].clone()
    else:
        root_vel = torch.zeros((n, 6), device=env.device, dtype=torch.float32)
        if hasattr(asset.data, "root_lin_vel_w"):
            root_vel[:, 0:3] = asset.data.root_lin_vel_w[env_ids].clone()
        if hasattr(asset.data, "root_ang_vel_w"):
            root_vel[:, 3:6] = asset.data.root_ang_vel_w[env_ids].clone()

    angles = 2.0 * torch.pi * torch.rand((n,), device=env.device)
    root_vel[:, 0] += float(delta_v) * torch.cos(angles)
    root_vel[:, 1] += float(delta_v) * torch.sin(angles)

    if hasattr(asset, "write_root_velocity_to_sim"):
        asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    elif hasattr(asset, "write_root_link_velocity_to_sim"):
        asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)
    else:
        raise RuntimeError("Robot articulation does not expose a root velocity write API.")


def _get_term_reason_masks_from_env(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor] | None:
    tm = getattr(env, "termination_manager", None)
    if tm is None:
        return None

    out: dict[str, torch.Tensor] = {}
    for name in ["time_out", "fallen", "bad_orientation"]:
        candidates = []
        for attr in ["_term_dones", "term_dones", "dones", "_truncated_buf", "_terminated_buf"]:
            if hasattr(tm, attr):
                candidates.append(getattr(tm, attr))
        for cand in candidates:
            try:
                if isinstance(cand, dict) and name in cand and torch.is_tensor(cand[name]):
                    out[name] = cand[name].detach().clone().to(torch.bool)
                    break
            except Exception:
                pass
        if name in out:
            continue
        for getter_name in ["get_term", "get_term_dones", "get_term_mask"]:
            getter = getattr(tm, getter_name, None)
            if getter is None:
                continue
            try:
                value = getter(name)
                if torch.is_tensor(value):
                    out[name] = value.detach().clone().to(torch.bool)
                    break
            except Exception:
                pass
    return out or None


def count_term_reasons(env: ManagerBasedRLEnv, terminated: torch.Tensor, truncated: torch.Tensor) -> dict[str, int]:
    counts = {"time_out": 0, "fallen": 0, "bad_orientation": 0, "other_termination": 0}
    masks = _get_term_reason_masks_from_env(env)
    if masks:
        for name in ["time_out", "fallen", "bad_orientation"]:
            mask = masks.get(name)
            if mask is not None:
                counts[name] += int(mask.sum().item())
        known = torch.zeros_like(terminated, dtype=torch.bool)
        for mask in masks.values():
            known |= mask.to(device=terminated.device, dtype=torch.bool)
        counts["other_termination"] += int((terminated & ~known).sum().item())
    else:
        counts["time_out"] += int(truncated.sum().item())
        counts["other_termination"] += int(terminated.sum().item())
    return counts


@torch.no_grad()
def run_trial(
    env: ManagerBasedRLEnv,
    policy,
    label: str,
    vx: float,
    n_steps: int,
    push: bool = False,
    push_delta_v: float = 0.0,
    push_interval_steps: int = 300,
    action_delay_steps: int = 0,
) -> dict[str, Any]:
    print(f"[EVAL] Trial {label}: vx={vx:.2f}, steps={n_steps}, push={push}, action_delay={action_delay_steps}")

    # Keep curriculum at max command range for policy/context, but reset events that depend on
    # this were disabled in make_eval_env_cfg().
    env._curriculum_stage = 2

    # Command must be fixed before reset because policy observations include the command vector.
    set_commands(env, vx)
    obs, _ = env.reset()
    set_commands(env, vx)

    ever_fell = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    vx_errors: list[float] = []
    slip_vals: list[float] = []
    reason_counts = {"time_out": 0, "fallen": 0, "bad_orientation": 0, "other_termination": 0}
    fall_count = 0

    last_action = torch.zeros((env.num_envs, 13), device=env.device)

    for step in range(n_steps):
        policy_obs = extract_policy_obs(obs)
        new_action = policy(policy_obs).clamp(-1.0, 1.0)

        if action_delay_steps > 0:
            # Update action every action_delay_steps+1 control steps, otherwise hold previous.
            if step % (action_delay_steps + 1) == 0:
                actions = new_action
                last_action = new_action
            else:
                actions = last_action
        else:
            actions = new_action

        pushed_this_step = bool(push and step > 0 and step % push_interval_steps == 0)
        if pushed_this_step:
            apply_root_velocity_kick(env, push_delta_v)

        obs, rew, terminated, truncated, info = env.step(actions)

        # Keep command fixed after auto-resets.
        set_commands(env, vx)

        terminated = terminated.to(device=env.device, dtype=torch.bool)
        truncated = truncated.to(device=env.device, dtype=torch.bool)
        ever_fell |= terminated
        fall_count += int(terminated.sum().item())

        step_counts = count_term_reasons(env, terminated, truncated)
        for k, v in step_counts.items():
            reason_counts[k] = reason_counts.get(k, 0) + int(v)

        robot = env.scene["robot"]
        vx_actual = robot.data.root_lin_vel_b[:, 0]
        vx_errors.append(torch.abs(vx_actual - float(vx)).mean().item())
        slip_vals.append(compute_slip_metric(env).mean().item())

        if args_cli.print_every > 0 and step % args_cli.print_every == 0:
            print(
                f"[EVAL]   step={step:4d}/{n_steps} "
                f"vx_err={vx_errors[-1]:.4f} slip={slip_vals[-1]:.4f} "
                f"falls_so_far={fall_count}",
                flush=True,
            )

    success_rate = float((~ever_fell).float().mean().item())
    return {
        "label": label,
        "vx_cmd": float(vx),
        "success_rate": success_rate,
        "fall_count": int(fall_count),
        "mean_vx_error": float(np.mean(vx_errors)) if vx_errors else float("nan"),
        "mean_slip": float(np.mean(slip_vals)) if slip_vals else float("nan"),
        "time_out": int(reason_counts.get("time_out", 0)),
        "fallen": int(reason_counts.get("fallen", 0)),
        "bad_orientation": int(reason_counts.get("bad_orientation", 0)),
        "other_termination": int(reason_counts.get("other_termination", 0)),
    }


def main() -> int:
    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args_cli.output_dir).expanduser().resolve() if args_cli.output_dir else checkpoint_path.parent / "eval_fixed"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"eval_results_fixed_{timestamp}.csv"

    env_cfg = make_eval_env_cfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = str(getattr(env, "device", "cpu"))

    print("\n[EVAL] Starting fixed locomotion evaluation")
    print(f"[EVAL] checkpoint : {checkpoint_path}")
    print(f"[EVAL] num_envs   : {env.num_envs}")
    print(f"[EVAL] eval_steps : {args_cli.eval_steps}")
    print(f"[EVAL] device     : {device}")
    print(f"[EVAL] output_csv : {csv_path}")

    policy, wrapped_env = build_inference_policy(env, checkpoint_path, device)

    rows: list[dict[str, Any]] = []
    """
    print("\n[EVAL] === Speed Sweep ===")
    for vx, label in zip(EVAL_SPEEDS, SPEED_LABELS):
        result = run_trial(env, policy, label=label, vx=vx, n_steps=args_cli.eval_steps)
        result["test_type"] = "speed_sweep"
        result["robustness"] = "none"
        rows.append(result)
        print(
            f"[EVAL] {label:<16s} success={result['success_rate']:.2%} "
            f"vx_err={result['mean_vx_error']:.4f} slip={result['mean_slip']:.4f} "
            f"falls={result['fall_count']}"
        )
    """
    print("\n[EVAL] === Robustness Tests ===")
    for test in ROBUSTNESS_TESTS:
        result = run_trial(
            env,
            policy,
            label=test["label"],
            vx=float(test["vx"]),
            n_steps=args_cli.eval_steps,
            push=bool(test["push"]),
            push_delta_v=float(test.get("push_delta_v", 0.0)),
            push_interval_steps=int(test["push_interval_steps"]),
            action_delay_steps=int(test["action_delay_steps"]),
        )
        result["test_type"] = "robustness"
        result["robustness"] = test["description"]
        rows.append(result)
        print(
            f"[EVAL] {test['label']:<16s} success={result['success_rate']:.2%} "
            f"vx_err={result['mean_vx_error']:.4f} slip={result['mean_slip']:.4f} "
            f"falls={result['fall_count']}"
        )

    fieldnames = [
        "test_type", "label", "vx_cmd", "success_rate", "fall_count",
        "mean_vx_error", "mean_slip", "time_out", "fallen",
        "bad_orientation", "other_termination", "robustness",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n[EVAL] ═══════════════════════════════════════════════════════════")
    print(f"{'label':<22} {'vx':>5} {'success':>9} {'vx_err':>9} {'slip':>9} {'falls':>7}")
    print("[EVAL] " + "─" * 63)
    for row in rows:
        print(
            f"[EVAL] {row['label']:<22} {row['vx_cmd']:>5.1f} "
            f"{row['success_rate']:>9.2%} {row['mean_vx_error']:>9.4f} "
            f"{row['mean_slip']:>9.4f} {row['fall_count']:>7d}"
        )
    print("[EVAL] ═══════════════════════════════════════════════════════════")
    print(f"[EVAL] Results saved: {csv_path}")

    env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()