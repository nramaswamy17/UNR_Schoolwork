#!/usr/bin/env python3
"""
Evaluate a trained PPO checkpoint for the N1 stand task.

What it reports:
- Standing-duration pass %  (survive >= pass_duration_s without falling)
- Success rate %            (no fall during the evaluation window)
- Termination breakdown     (time_out / fallen / bad_orientation / other)
- Optional hard-reset robustness test

This script is intentionally defensive because Isaac Lab / RSL-RL APIs vary by version.
It tries multiple import and checkpoint-loading paths before failing.

Examples
--------
Nominal evaluation:
    ./isaaclab.sh -p evaluate_n1_stand.py \
        --checkpoint /path/to/model_400.pt \
        --headless --num_envs 64 --num_episodes 50

Hard-reset robustness evaluation:
    ./isaaclab.sh -p evaluate_n1_stand.py \
        --checkpoint /path/to/model_400.pt \
        --headless --num_envs 64 --num_episodes 50 \
        --robustness hard_reset --hard_reset_roll_pitch_deg 6.0 --hard_reset_yaw_deg 20.0
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate PPO checkpoint on the N1 stand task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint, e.g. model_400.pt")
parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments for evaluation")
parser.add_argument("--num_episodes", type=int, default=50, help="Total episodes to evaluate")
parser.add_argument("--episode_length_s", type=float, default=20.0, help="Evaluation episode length in seconds")
parser.add_argument("--pass_duration_s", type=float, default=20.0, help="Seconds required to count as a pass")
parser.add_argument(
    "--robustness",
    type=str,
    default="none",
    choices=["none", "hard_reset"],
    help="Robustness mode. 'hard_reset' increases reset orientation randomization.",
)
parser.add_argument("--hard_reset_roll_pitch_deg", type=float, default=6.0, help="Hard reset roll/pitch randomization in degrees")
parser.add_argument("--hard_reset_yaw_deg", type=float, default=20.0, help="Hard reset yaw randomization in degrees")
parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
parser.add_argument(
    "--results_dir",
    type=str,
    default=None,
    help="Directory for CSV/text outputs. Defaults to <checkpoint_dir>/evaluation_<timestamp>",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Post-launch imports
# -----------------------------------------------------------------------------
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from isaaclab.envs import ManagerBasedRLEnv

try:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
except Exception:
    RslRlVecEnvWrapper = None

try:
    from rsl_rl.runners import OnPolicyRunner
except Exception:
    try:
        from rsl_rl.runners.on_policy_runner import OnPolicyRunner
    except Exception:
        OnPolicyRunner = None

try:
    from isaaclab.utils.dict import class_to_dict
except Exception:
    class_to_dict = None

from humanoid_n1_env_cfg_updated import HumanoidN1EnvCfg
from agents.rsl_rl_ppo_cfg import HumanoidN1PPORunnerCfg


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def cfg_to_dict(cfg_obj):
    if hasattr(cfg_obj, "to_dict"):
        return cfg_obj.to_dict()
    if class_to_dict is not None:
        return class_to_dict(cfg_obj)
    if isinstance(cfg_obj, dict):
        return cfg_obj
    raise TypeError(f"Could not convert config object of type {type(cfg_obj)} to dict.")


def ensure_results_dir(checkpoint_path: Path, results_dir_arg: str | None) -> Path:
    if results_dir_arg is not None:
        out_dir = Path(results_dir_arg).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = checkpoint_path.resolve().parent / f"evaluation_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def set_hard_reset(cfg: HumanoidN1EnvCfg, roll_pitch_deg: float, yaw_deg: float) -> None:
    """Best-effort patch of the reset randomization parameters."""
    events = getattr(cfg, "events", None)
    if events is None:
        return
    term = getattr(events, "randomize_root_pose", None)
    if term is None:
        return

    params = dict(getattr(term, "params", {}) or {})
    params["roll_pitch_deg"] = float(roll_pitch_deg)
    params["yaw_deg"] = float(yaw_deg)
    try:
        term.params = params
    except Exception:
        pass


@torch.no_grad()
def build_inference_policy(eval_env, checkpoint_path: Path, device: str):
    """Load the trained actor and return a callable policy(obs_tensor)->actions."""
    if OnPolicyRunner is None:
        raise ImportError("Could not import RSL-RL OnPolicyRunner. Use the same Isaac Lab env as training.")
    if RslRlVecEnvWrapper is None:
        raise ImportError("Could not import isaaclab_rl.rsl_rl.RslRlVecEnvWrapper.")

    wrapped_env = RslRlVecEnvWrapper(eval_env)
    agent_cfg = HumanoidN1PPORunnerCfg()
    train_cfg_dict = cfg_to_dict(agent_cfg)

    runner = OnPolicyRunner(
        env=wrapped_env,
        train_cfg=train_cfg_dict,
        log_dir=None,
        device=str(device),
    )

    # Try the common checkpoint load path first.
    loaded = False
    load_errors: list[str] = []

    for attempt in [
        lambda: runner.load(str(checkpoint_path)),
        lambda: runner.load(checkpoint_path),
    ]:
        try:
            attempt()
            loaded = True
            break
        except Exception as exc:
            load_errors.append(repr(exc))

    # Fallback: raw torch state dict loading into actor_critic if runner.load() path differs.
    if not loaded:
        ckpt = torch.load(str(checkpoint_path), map_location=device)
        actor_critic = None
        for attr_chain in [
            ("alg", "actor_critic"),
            ("algorithm", "actor_critic"),
            ("actor_critic",),
        ]:
            obj = runner
            ok = True
            for attr in attr_chain:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok:
                actor_critic = obj
                break

        if actor_critic is None:
            raise RuntimeError(
                "Could not find actor_critic on runner, and runner.load() also failed.\n"
                + "\n".join(load_errors)
            )

        state_keys = ["model_state_dict", "state_dict", "actor_critic_state_dict"]
        state_dict = None
        if isinstance(ckpt, dict):
            for k in state_keys:
                if k in ckpt:
                    state_dict = ckpt[k]
                    break
            if state_dict is None and all(isinstance(k, str) for k in ckpt.keys()):
                state_dict = ckpt
        else:
            state_dict = ckpt

        if state_dict is None:
            raise RuntimeError(
                "Checkpoint format not recognized. runner.load() failed and no obvious state_dict was found.\n"
                + "\n".join(load_errors)
            )
        actor_critic.load_state_dict(state_dict, strict=False)

    # Common inference-policy accessors across versions.
    for getter in [
        lambda: runner.get_inference_policy(device=device),
        lambda: runner.get_inference_policy(),
    ]:
        try:
            policy = getter()
            if callable(policy):
                return policy
        except Exception:
            pass

    # Final fallback: direct actor_critic inference.
    actor_critic = None
    for attr_chain in [
        ("alg", "actor_critic"),
        ("algorithm", "actor_critic"),
        ("actor_critic",),
    ]:
        obj = runner
        ok = True
        for attr in attr_chain:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok:
            actor_critic = obj
            break
    if actor_critic is None:
        raise RuntimeError("Checkpoint loaded, but could not recover an inference policy.")

    def _policy(obs_tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(actor_critic, "act_inference"):
            return actor_critic.act_inference(obs_tensor)
        if hasattr(actor_critic, "act"):
            out = actor_critic.act(obs_tensor)
            if isinstance(out, tuple):
                return out[0]
            return out
        raise RuntimeError("actor_critic has neither act_inference() nor act().")

    return _policy



def extract_policy_obs(obs: Any) -> torch.Tensor:
    if torch.is_tensor(obs):
        return obs
    if isinstance(obs, dict):
        if "policy" in obs and torch.is_tensor(obs["policy"]):
            return obs["policy"]
        # defensive fallback for other layouts
        for key in ["obs", "observation", "actor"]:
            if key in obs and torch.is_tensor(obs[key]):
                return obs[key]
    raise TypeError(f"Could not extract policy observation from type {type(obs)}")


@dataclass
class EpisodeRecord:
    episode_id: int
    env_slot: int
    mode: str
    survived_s: float
    success: int
    standing_duration_pass: int
    termination_reason: str


class EpisodeTracker:
    def __init__(self, num_envs: int, episode_dt: float, pass_duration_s: float):
        self.num_envs = int(num_envs)
        self.episode_dt = float(episode_dt)
        self.pass_duration_s = float(pass_duration_s)
        self.elapsed_s = torch.zeros(self.num_envs, dtype=torch.float32)
        self.started = torch.zeros(self.num_envs, dtype=torch.bool)
        self.episode_ids = [-1 for _ in range(self.num_envs)]
        self.records: list[EpisodeRecord] = []
        self.next_episode_id = 0

    def start_initial(self, active_envs: int):
        for i in range(active_envs):
            self.started[i] = True
            self.episode_ids[i] = self.next_episode_id
            self.next_episode_id += 1

    def start_replacements(self, done_indices: list[int], total_target: int):
        for idx in done_indices:
            if self.next_episode_id >= total_target:
                self.started[idx] = False
                self.episode_ids[idx] = -1
                self.elapsed_s[idx] = 0.0
                continue
            self.started[idx] = True
            self.episode_ids[idx] = self.next_episode_id
            self.elapsed_s[idx] = 0.0
            self.next_episode_id += 1

    def step_active(self):
        self.elapsed_s[self.started] += self.episode_dt

    def finalize_done(
        self,
        done_mask: torch.Tensor,
        truncated_mask: torch.Tensor,
        reason_masks: dict[str, torch.Tensor] | None,
        mode: str,
    ) -> list[int]:
        done_indices = torch.nonzero(done_mask, as_tuple=False).flatten().tolist()
        for idx in done_indices:
            ep_id = self.episode_ids[idx]
            survived_s = float(self.elapsed_s[idx].item())
            term_reason = infer_reason_for_env(idx, bool(truncated_mask[idx].item()), reason_masks)
            success = int(term_reason == "time_out" and survived_s >= self.pass_duration_s)
            standing_pass = int(success == 1)
            self.records.append(
                EpisodeRecord(
                    episode_id=int(ep_id),
                    env_slot=int(idx),
                    mode=mode,
                    survived_s=survived_s,
                    success=success,
                    standing_duration_pass=standing_pass,
                    termination_reason=term_reason,
                )
            )
            self.elapsed_s[idx] = 0.0
        return done_indices



def _get_term_reason_masks_from_env(env) -> dict[str, torch.Tensor] | None:
    """Best-effort access to per-env termination masks.

    Isaac Lab internals differ across versions. We try several likely layouts.
    If nothing works, caller falls back to time_out vs other.
    """
    tm = getattr(env, "termination_manager", None)
    if tm is None:
        return None

    out: dict[str, torch.Tensor] = {}
    for name in ["time_out", "fallen", "bad_orientation"]:
        candidates = []
        for attr in ["_term_dones", "term_dones", "dones", "_truncated_buf", "_terminated_buf"]:
            if hasattr(tm, attr):
                candidates.append(getattr(tm, attr))
        # dict-like container
        for cand in candidates:
            try:
                if isinstance(cand, dict) and name in cand and torch.is_tensor(cand[name]):
                    out[name] = cand[name].detach().clone().to(torch.bool).cpu()
                    break
            except Exception:
                pass
        if name in out:
            continue
        # some versions expose dedicated getter methods
        for getter_name in ["get_term", "get_term_dones", "get_term_mask"]:
            getter = getattr(tm, getter_name, None)
            if getter is None:
                continue
            try:
                value = getter(name)
                if torch.is_tensor(value):
                    out[name] = value.detach().clone().to(torch.bool).cpu()
                    break
            except Exception:
                pass
    return out or None



def infer_reason_for_env(env_idx: int, is_truncated: bool, reason_masks: dict[str, torch.Tensor] | None) -> str:
    if reason_masks is not None:
        for name in ["time_out", "fallen", "bad_orientation"]:
            mask = reason_masks.get(name)
            if mask is not None and env_idx < mask.numel() and bool(mask[env_idx].item()):
                return name
    if is_truncated:
        return "time_out"
    return "other_termination"



def summarize_records(records: list[EpisodeRecord], pass_duration_s: float) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        raise RuntimeError("No completed evaluation episodes were recorded.")

    passed = sum(r.standing_duration_pass for r in records)
    successes = sum(r.success for r in records)
    mean_survival_s = sum(r.survived_s for r in records) / total

    term_counts: dict[str, int] = {}
    for r in records:
        term_counts[r.termination_reason] = term_counts.get(r.termination_reason, 0) + 1

    return {
        "episodes": total,
        "pass_duration_s": float(pass_duration_s),
        "passes": passed,
        "standing_duration_pass_pct": 100.0 * passed / total,
        "successes": successes,
        "success_rate_pct": 100.0 * successes / total,
        "mean_survival_s": mean_survival_s,
        "termination_counts": term_counts,
    }



def write_episode_csv(path: Path, records: list[EpisodeRecord]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_id",
                "env_slot",
                "mode",
                "survived_s",
                "success",
                "standing_duration_pass",
                "termination_reason",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "episode_id": r.episode_id,
                "env_slot": r.env_slot,
                "mode": r.mode,
                "survived_s": f"{r.survived_s:.4f}",
                "success": r.success,
                "standing_duration_pass": r.standing_duration_pass,
                "termination_reason": r.termination_reason,
            })



def write_summary_txt(path: Path, summary: dict[str, Any], mode: str) -> None:
    tc = summary["termination_counts"]
    with open(path, "w") as f:
        f.write(f"Evaluation mode: {mode}\n")
        f.write(f"Episodes: {summary['episodes']}\n")
        f.write(f"Pass duration threshold (s): {summary['pass_duration_s']:.2f}\n")
        f.write(f"Mean survival time (s): {summary['mean_survival_s']:.3f}\n")
        f.write(f"Standing-duration pass: {summary['passes']}/{summary['episodes']} = {summary['standing_duration_pass_pct']:.2f}%\n")
        f.write(f"Success rate: {summary['successes']}/{summary['episodes']} = {summary['success_rate_pct']:.2f}%\n")
        f.write("Termination counts:\n")
        for name, count in sorted(tc.items()):
            f.write(f"  {name}: {count}\n")


@torch.no_grad()
def evaluate_once() -> int:
    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args_cli.num_episodes <= 0:
        raise ValueError("--num_episodes must be positive")
    if args_cli.num_envs <= 0:
        raise ValueError("--num_envs must be positive")

    results_dir = ensure_results_dir(checkpoint_path, args_cli.results_dir)

    env_cfg = HumanoidN1EnvCfg()
    env_cfg.scene.num_envs = int(min(args_cli.num_envs, args_cli.num_episodes))
    env_cfg.episode_length_s = float(args_cli.episode_length_s)
    if args_cli.seed is not None:
        try:
            env_cfg.seed = int(args_cli.seed)
        except Exception:
            pass

    if args_cli.robustness == "hard_reset":
        set_hard_reset(
            env_cfg,
            roll_pitch_deg=float(args_cli.hard_reset_roll_pitch_deg),
            yaw_deg=float(args_cli.hard_reset_yaw_deg),
        )

    env = ManagerBasedRLEnv(cfg=env_cfg)

    device = str(getattr(env, "device", "cpu"))
    step_dt = float(getattr(env, "step_dt", getattr(env_cfg.sim, "dt", 1.0) * getattr(env_cfg, "decimation", 1)))

    print("\n[EVAL] Starting evaluation")
    print(f"[EVAL] checkpoint          : {checkpoint_path}")
    print(f"[EVAL] mode                : {args_cli.robustness}")
    print(f"[EVAL] num_envs            : {env.num_envs}")
    print(f"[EVAL] num_episodes        : {args_cli.num_episodes}")
    print(f"[EVAL] episode_length_s    : {env_cfg.episode_length_s}")
    print(f"[EVAL] pass_duration_s     : {args_cli.pass_duration_s}")
    print(f"[EVAL] step_dt             : {step_dt}")
    print(f"[EVAL] device              : {device}")
    print(f"[EVAL] results_dir         : {results_dir}")

    policy = build_inference_policy(env, checkpoint_path, device)

    obs, _ = env.reset()
    tracker = EpisodeTracker(num_envs=env.num_envs, episode_dt=step_dt, pass_duration_s=float(args_cli.pass_duration_s))
    tracker.start_initial(active_envs=env.num_envs)

    last_print_completed = -1
    while len(tracker.records) < args_cli.num_episodes:
        policy_obs = extract_policy_obs(obs)
        actions = policy(policy_obs)
        obs, reward, terminated, truncated, info = env.step(actions)

        # Increase elapsed time for currently active episodes.
        tracker.step_active()

        terminated_cpu = terminated.detach().to(torch.bool).cpu()
        truncated_cpu = truncated.detach().to(torch.bool).cpu()
        done_cpu = torch.logical_or(terminated_cpu, truncated_cpu)

        # Best-effort fine-grained reason masks.
        reason_masks = _get_term_reason_masks_from_env(env)

        done_indices = tracker.finalize_done(
            done_mask=done_cpu,
            truncated_mask=truncated_cpu,
            reason_masks=reason_masks,
            mode=args_cli.robustness,
        )
        tracker.start_replacements(done_indices=done_indices, total_target=int(args_cli.num_episodes))

        completed = len(tracker.records)
        if completed != last_print_completed and (completed % 10 == 0 or completed == args_cli.num_episodes):
            print(f"[EVAL] completed_episodes  : {completed}/{args_cli.num_episodes}")
            last_print_completed = completed

    # Keep only the requested count in case multiple envs finish on the last step.
    records = sorted(tracker.records, key=lambda r: r.episode_id)[: args_cli.num_episodes]
    summary = summarize_records(records, pass_duration_s=float(args_cli.pass_duration_s))

    episode_csv = results_dir / "evaluation_episodes.csv"
    summary_txt = results_dir / "evaluation_summary.txt"
    write_episode_csv(episode_csv, records)
    write_summary_txt(summary_txt, summary, mode=args_cli.robustness)

    tc = summary["termination_counts"]
    print("\n[EVAL] Done")
    print(f"[EVAL] Mean survival time (s)      : {summary['mean_survival_s']:.3f}")
    print(f"[EVAL] Standing-duration pass %    : {summary['standing_duration_pass_pct']:.2f}")
    print(f"[EVAL] Success rate %              : {summary['success_rate_pct']:.2f}")
    print(f"[EVAL] Termination counts          : {tc}")
    print(f"[EVAL] Episode CSV                 : {episode_csv}")
    print(f"[EVAL] Summary TXT                 : {summary_txt}")

    env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(evaluate_once())
    finally:
        simulation_app.close()
