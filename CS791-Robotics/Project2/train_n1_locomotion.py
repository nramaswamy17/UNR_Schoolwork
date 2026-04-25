# train_n1_locomotion.py
#
# CS791 Project 2 – PPO training for walk / run locomotion.
#
# Curriculum is time-based (iteration checkpoints) rather than
# reward-based, which is more reproducible and aligns with the
# project report's described schedule.
#
# Usage (from humanoid_n1_task root, via isaaclab.sh wrapper):
#
#   ./isaaclab.sh -p train_n1_locomotion.py \
#       --num_envs 512 --max_iterations 3000 --headless

import os
import sys
import argparse
from datetime import datetime

from isaaclab.app import AppLauncher

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train PPO for N1 Walk/Run (Project 2).")
parser.add_argument("--num_envs",       type=int,   default=None)
parser.add_argument("--max_iterations", type=int,   default=3000)
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--experiment_name",type=str,   default=None)
parser.add_argument("--run_name",       type=str,   default=None)
parser.add_argument("--log_root",       type=str,   default=None)
parser.add_argument("--episode_length_s", type=float, default=15.0)
# Curriculum stage thresholds (iterations)
parser.add_argument("--stage1_iter", type=int, default=600,
                    help="Iteration at which to advance to curriculum stage 1 (walk).")
parser.add_argument("--stage2_iter", type=int, default=1400,
                    help="Iteration at which to advance to curriculum stage 2 (run).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ──────────────────────────────────────────────────────
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from humanoid_n1_locomotion_env_cfg_v2 import (
    HumanoidN1LocomotionEnvCfgV2,
    CURRICULUM_STAGES,
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


# ── Helpers ──────────────────────────────────────────────────────────────────
def cfg_to_dict(cfg_obj):
    if hasattr(cfg_obj, "to_dict"):       return cfg_obj.to_dict()
    if class_to_dict is not None:         return class_to_dict(cfg_obj)
    if isinstance(cfg_obj, dict):         return cfg_obj
    raise TypeError(f"Cannot convert {type(cfg_obj)} to dict.")


def make_log_dir(agent_cfg) -> str:
    root            = args_cli.log_root or os.path.join(_THIS_DIR, "logs")
    experiment_name = args_cli.experiment_name or getattr(agent_cfg, "experiment_name", "humanoid_n1_locomotion")
    run_name        = args_cli.run_name        or getattr(agent_cfg, "run_name",        "ppo_walk_run")
    timestamp       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir         = os.path.join(root, experiment_name, f"{run_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def build_env_cfg() -> HumanoidN1LocomotionEnvCfgV2:
    cfg = HumanoidN1LocomotionEnvCfgV2()
    if args_cli.num_envs is not None:
        cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.episode_length_s = float(args_cli.episode_length_s)
    try:
        cfg.viewer.eye = (12.0, 12.0, 7.0)
    except Exception:
        pass
    return cfg


def build_agent_cfg() -> HumanoidN1LocomotionPPORunnerCfg:
    cfg = HumanoidN1LocomotionPPORunnerCfg()
    if args_cli.seed           is not None: cfg.seed           = int(args_cli.seed)
    if args_cli.max_iterations is not None: cfg.max_iterations = int(args_cli.max_iterations)
    if args_cli.experiment_name is not None: cfg.experiment_name = str(args_cli.experiment_name)
    if args_cli.run_name        is not None: cfg.run_name        = str(args_cli.run_name)
    return cfg


STAGE_NAMES = ["Stage 0 – Slow Walk (0–0.8 m/s)",
               "Stage 1 – Walk     (0–1.8 m/s)",
               "Stage 2 – Run      (0–3.2 m/s)"]


class CurriculumCallback:
    """
    Checks current iteration and advances curriculum stage when thresholds
    are reached.  Meant to be called from the training loop.
    """
    def __init__(self, env_unwrapped, stage1_iter: int, stage2_iter: int):
        self.env        = env_unwrapped
        self.thresholds = [stage1_iter, stage2_iter]
        self._stage     = 0
        env_unwrapped._curriculum_stage = 0
        print(f"[CURRICULUM] Initialized at {STAGE_NAMES[0]}")

    def step(self, iteration: int) -> None:
        if self._stage < len(self.thresholds):
            if iteration >= self.thresholds[self._stage]:
                self._stage += 1
                self.env._curriculum_stage = self._stage
                print(f"[CURRICULUM] iter={iteration}: advanced to {STAGE_NAMES[self._stage]}")
                stage_cfg = CURRICULUM_STAGES[self._stage]
                print(f"[CURRICULUM]   vx_max={stage_cfg['vx_max']} "
                      f"vy_max={stage_cfg['vy_max']} "
                      f"push={stage_cfg['push']}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    env_cfg   = build_env_cfg()
    agent_cfg = build_agent_cfg()
    log_dir   = make_log_dir(agent_cfg)

    print("\n[TRAIN] CS791 Project 2 – Humanoid Walk/Run PPO")
    print(f"[TRAIN] num_envs          : {env_cfg.scene.num_envs}")
    print(f"[TRAIN] episode_length_s  : {env_cfg.episode_length_s}")
    print(f"[TRAIN] max_iterations    : {agent_cfg.max_iterations}")
    print(f"[TRAIN] stage1_iter       : {args_cli.stage1_iter}")
    print(f"[TRAIN] stage2_iter       : {args_cli.stage2_iter}")
    print(f"[TRAIN] seed              : {agent_cfg.seed}")
    print(f"[TRAIN] log_dir           : {log_dir}")

    # Create environment
    env_base = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"[TRAIN] action_space      : {env_base.action_space}")

    obs, _ = env_base.reset()
    print(f"[TRAIN] obs keys          : {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")

    # Curriculum callback
    curriculum = CurriculumCallback(
        env_base,
        stage1_iter=args_cli.stage1_iter,
        stage2_iter=args_cli.stage2_iter,
    )

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env_base)

    # Build runner
    train_cfg_dict = cfg_to_dict(agent_cfg)
    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg_dict,
        log_dir=log_dir,
        device=str(agent_cfg.device),
    )

    # ── Training loop with curriculum callbacks ───────────────────────────
    # RSL-RL's OnPolicyRunner.learn() doesn't expose per-iteration hooks,
    # so we call it in short chunks to insert curriculum updates.
    CHUNK = 50   # advance curriculum check every 50 iterations
    total = int(agent_cfg.max_iterations)
    done  = 0

    while done < total:
        chunk_size = min(CHUNK, total - done)
        runner.learn(num_learning_iterations=chunk_size, init_at_random_ep_len=(done == 0))
        done += chunk_size
        curriculum.step(done)

    print(f"\n[TRAIN] Training complete. Logs: {log_dir}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
