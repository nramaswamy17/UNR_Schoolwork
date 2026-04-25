# train_n1_stand.py
#
# PPO training launcher for the N1 stand task.
# Uses direct local imports to avoid package-registration/import-path issues.

import os
import sys
import argparse
from datetime import datetime

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train PPO on the N1 stand task.")

# Custom args first
parser.add_argument("--num_envs", type=int, default=None, help="Override env_cfg.scene.num_envs")
parser.add_argument("--max_iterations", type=int, default=None, help="Override PPO max_iterations")
parser.add_argument("--seed", type=int, default=None, help="Override PPO seed")
parser.add_argument("--experiment_name", type=str, default=None, help="Override PPO experiment_name")
parser.add_argument("--run_name", type=str, default=None, help="Override PPO run_name")
parser.add_argument("--log_root", type=str, default=None, help="Root directory for logs/checkpoints")
parser.add_argument("--episode_length_s", type=float, default=20.0, help="Episode length in seconds")

# Isaac Lab app args second
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Post-launch imports
# -----------------------------------------------------------------------------
import torch

# Make sure the script directory itself is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from humanoid_n1_env_cfg_updated import HumanoidN1EnvCfg
from agents.rsl_rl_ppo_cfg import HumanoidN1PPORunnerCfg

# Try the most likely runner imports for Isaac Lab / RSL-RL versions
_ONPOLICY_IMPORT_ERROR = None
try:
    from rsl_rl.runners import OnPolicyRunner
except Exception as exc_1:
    try:
        from rsl_rl.runners.on_policy_runner import OnPolicyRunner
    except Exception as exc_2:
        _ONPOLICY_IMPORT_ERROR = (exc_1, exc_2)
        OnPolicyRunner = None

# Optional helper for config conversion
try:
    from isaaclab.utils.dict import class_to_dict
except Exception:
    class_to_dict = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def cfg_to_dict(cfg_obj):
    """Convert Isaac Lab configclass objects to plain dicts."""
    if hasattr(cfg_obj, "to_dict"):
        return cfg_obj.to_dict()
    if class_to_dict is not None:
        return class_to_dict(cfg_obj)
    if isinstance(cfg_obj, dict):
        return cfg_obj
    raise TypeError(
        f"Could not convert config object of type {type(cfg_obj)} to dict. "
        "Expected .to_dict() or isaaclab.utils.dict.class_to_dict."
    )


def make_log_dir(agent_cfg) -> str:
    root = args_cli.log_root
    if root is None:
        root = os.path.join(_THIS_DIR, "logs")

    experiment_name = args_cli.experiment_name or getattr(agent_cfg, "experiment_name", "humanoid_n1")
    run_name = args_cli.run_name or getattr(agent_cfg, "run_name", "ppo_n1")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_dir = os.path.join(root, experiment_name, f"{run_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def build_env_cfg():
    env_cfg = HumanoidN1EnvCfg()

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = int(args_cli.num_envs)

    # Project requirement is 20s standing; default this launcher to 20s.
    env_cfg.episode_length_s = float(args_cli.episode_length_s)

    # Nice default camera if viewer is enabled
    if hasattr(env_cfg, "viewer"):
        try:
            env_cfg.viewer.eye = (12.0, 12.0, 7.0)
        except Exception:
            pass

    return env_cfg


def build_agent_cfg():
    agent_cfg = HumanoidN1PPORunnerCfg()

    if args_cli.seed is not None:
        agent_cfg.seed = int(args_cli.seed)
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = int(args_cli.max_iterations)
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = str(args_cli.experiment_name)
    if args_cli.run_name is not None:
        agent_cfg.run_name = str(args_cli.run_name)

    return agent_cfg


def print_startup_summary(env_cfg, agent_cfg, log_dir):
    print("\n[TRAIN] Starting PPO training")
    print(f"[TRAIN] num_envs           : {env_cfg.scene.num_envs}")
    print(f"[TRAIN] episode_length_s  : {env_cfg.episode_length_s}")
    print(f"[TRAIN] seed               : {agent_cfg.seed}")
    print(f"[TRAIN] max_iterations     : {agent_cfg.max_iterations}")
    print(f"[TRAIN] num_steps_per_env  : {agent_cfg.num_steps_per_env}")
    print(f"[TRAIN] experiment_name    : {agent_cfg.experiment_name}")
    print(f"[TRAIN] run_name           : {agent_cfg.run_name}")
    print(f"[TRAIN] device             : {agent_cfg.device}")
    print(f"[TRAIN] headless           : {getattr(args_cli, 'headless', False)}")
    print(f"[TRAIN] log_dir            : {log_dir}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    if OnPolicyRunner is None:
        err_1, err_2 = _ONPOLICY_IMPORT_ERROR
        raise ImportError(
            "Could not import RSL-RL OnPolicyRunner.\n"
            f"First attempt error: {repr(err_1)}\n"
            f"Second attempt error: {repr(err_2)}"
        )

    env_cfg = build_env_cfg()
    agent_cfg = build_agent_cfg()
    log_dir = make_log_dir(agent_cfg)

    print_startup_summary(env_cfg, agent_cfg, log_dir)

    # Create env directly from local cfg, matching your working debug pattern
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"[TRAIN] env.device         : {env.device}")
    print(f"[TRAIN] env.num_envs       : {env.num_envs}")
    print(f"[TRAIN] action_space       : {env.action_space}")

    # Reset once up front so early env/config failures surface immediately
    obs, extras = env.reset()
    if isinstance(obs, dict):
        print(f"[TRAIN] obs keys           : {list(obs.keys())}")
        for k, v in obs.items():
            if torch.is_tensor(v):
                print(f"[TRAIN]   obs[{k}] shape   : {tuple(v.shape)}")
    else:
        print(f"[TRAIN] obs type           : {type(obs)}")

    # Wrap env for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Build train cfg dict
    train_cfg_dict = cfg_to_dict(agent_cfg)

    # Runner
    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg_dict,
        log_dir=log_dir,
        device=str(agent_cfg.device),
    )

    # Train
    runner.learn(
        num_learning_iterations=int(agent_cfg.max_iterations),
        init_at_random_ep_len=True,
    )

    print("\n[TRAIN] Training complete.")
    print(f"[TRAIN] Logs/checkpoints saved under: {log_dir}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
