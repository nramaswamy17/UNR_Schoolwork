# run_n1_stand.py
# Simple runner for HumanoidN1 stand environment

import torch
from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# Imports that require IsaacSim
from isaaclab.envs import ManagerBasedRLEnv
from humanoid_n1_env_cfg_updated import HumanoidN1EnvCfg





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

    # Create environment
    env_cfg = HumanoidN1EnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)

    print("\nEnvironment created.")
    print("num_envs:", env.num_envs)
    print("device:", env.device)

    print("action_space:", env.action_space)

    # Reset environment
    obs, _ = env.reset()

    print("\nObservation keys:", obs.keys())
    for k, v in obs.items():
        print(f"{k} shape:", v.shape)

    # Step environment
    num_steps = 300

    for step in range(num_steps):

        actions = torch.randn(
            (env.num_envs, env.action_space.shape[-1]),
            device=env.device
        )

        obs, reward, terminated, truncated, info = env.step(actions)

        if step % 50 == 0:
            print(
                f"step={step:4d} "
                f"reward_mean={reward.mean().item():.4f} "
                f"terminated={terminated.sum().item()} "
                f"truncated={truncated.sum().item()}"
            )
            _print_reward_breakdown(env, prefix="  ")

    print("\nFinished stepping environment.")

    env.close()


if __name__ == "__main__":
    main()
