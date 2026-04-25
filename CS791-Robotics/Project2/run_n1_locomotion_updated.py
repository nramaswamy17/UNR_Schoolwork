# run_n1_locomotion.py
# Simple debug runner for the N1 locomotion environment

import argparse
import torch
from isaaclab.app import AppLauncher


# CLI
parser = argparse.ArgumentParser(description="Run N1 locomotion env with random actions.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--steps", type=int, default=400, help="Number of simulation steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac imports after app launch
from isaaclab.envs import ManagerBasedRLEnv
from humanoid_n1_locomotion_env_cfg_updated import HumanoidN1LocomotionEnvCfg





def _iter_reward_terms(rewards_cfg):
    seen = set()
    for name, term in vars(rewards_cfg).items():
        if name.startswith("_"):
            continue
        if hasattr(term, "func") and hasattr(term, "weight"):
            seen.add(name)
            yield name, term
    for name in dir(rewards_cfg):
        if name.startswith("_") or name in seen:
            continue
        term = getattr(rewards_cfg, name)
        if hasattr(term, "func") and hasattr(term, "weight"):
            yield name, term



def _compute_reward_breakdown(env):
    rows = []
    rewards_cfg = getattr(env.cfg, "rewards", None)
    if rewards_cfg is None:
        return rows

    for name, term in _iter_reward_terms(rewards_cfg):
        try:
            params = dict(getattr(term, "params", {}) or {})
            raw = term.func(env, **params)
            if raw.ndim > 1:
                raw = raw.reshape(raw.shape[0], -1).mean(dim=1)
            weight = float(term.weight)
            raw_mean = raw.mean().item()
            contrib_mean = (weight * raw).mean().item()
            step_dt = float(getattr(env, "step_dt", getattr(env.cfg.sim, "dt", 1.0) * getattr(env.cfg, "decimation", 1)))
            scaled_contrib_mean = (step_dt * weight * raw).mean().item()
            rows.append((name, weight, raw_mean, contrib_mean, scaled_contrib_mean, None))
        except Exception as exc:
            rows.append((name, float(getattr(term, "weight", 0.0)), float("nan"), float("nan"), float("nan"), str(exc)))
    return rows



def _print_reward_breakdown(env, prefix=""):
    rows = _compute_reward_breakdown(env)
    if not rows:
        print(f"{prefix}reward breakdown unavailable")
        return
    print(f"{prefix}reward breakdown (mean raw -> mean weighted contribution):")
    for name, weight, raw_mean, contrib_mean, scaled_contrib_mean, err in rows:
        if err is None:
            print(
                f"{prefix}  {name:<16s} weight={weight:>7.3f} "
                f"raw_mean={raw_mean:>9.4f} contrib_mean={contrib_mean:>10.4f} "
                f"step_scaled={scaled_contrib_mean:>10.4f}"
            )
        else:
            print(f"{prefix}  {name:<16s} weight={weight:>7.3f} unavailable ({err})")

def main():
    env_cfg = HumanoidN1LocomotionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 2.5
    env_cfg.viewer.eye = (12.0, 12.0, 7.0)

    env = ManagerBasedRLEnv(cfg=env_cfg)

    print("\n[DEBUG] Environment created.")
    print(f"[DEBUG] num_envs={env.num_envs} device={env.device}")
    print(f"[DEBUG] action_space={env.action_space}")

    obs, _ = env.reset()
    print("[DEBUG] reset ok")

    print("\n[DEBUG] Observation summary:")
    print("  keys:", list(obs.keys()) if isinstance(obs, dict) else type(obs))
    if isinstance(obs, dict):
        for k, v in obs.items():
            print(f"  - {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    for step in range(args_cli.steps):
        actions = 2.0 * torch.rand((env.num_envs, 13), device=env.device) - 1.0
        obs, rew, terminated, truncated, info = env.step(actions)

        if step == 0:
            print(
                f"\n[DEBUG] action tensor shape={tuple(actions.shape)} "
                f"dtype={actions.dtype} min={actions.min().item():.4f} max={actions.max().item():.4f}"
            )

        if step % 50 == 0:
            robot = env.scene["robot"]

            if hasattr(env, "_n1_commands"):
                vx_cmd_mean = env._n1_commands[:, 0].mean().item()
            else:
                vx_cmd_mean = float("nan")

            if hasattr(robot.data, "root_lin_vel_b"):
                vx_mean = robot.data.root_lin_vel_b[:, 0].mean().item()
            else:
                vx_mean = float("nan")

            print(
                f"[DEBUG] step={step:4d} "
                f"rew_mean={rew.mean().item():.4f} "
                f"vx_cmd_mean={vx_cmd_mean:.4f} "
                f"vx_mean={vx_mean:.4f} "
                f"terminated={terminated.sum().item()} "
                f"truncated={truncated.sum().item()}"
            )
            _print_reward_breakdown(env, prefix="  [DEBUG] ")

    print("\n[DEBUG] Finished stepping environment.")
    env.close()


if __name__ == "__main__":
    main()
