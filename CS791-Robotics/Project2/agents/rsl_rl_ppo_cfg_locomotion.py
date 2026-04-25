# agents/rsl_rl_ppo_cfg_locomotion.py
#
# CS791 Project 2 – PPO Configuration for Walk/Run Task
#
# Based on the working Project 1 rsl_rl_ppo_cfg.py with these locomotion
# adjustments:
#   - max_iterations: 400 → 3000  (locomotion needs much more training)
#   - save_interval:  10  → 50    (less frequent saves for long runs)
#   - init_noise_std: 0.5 → 0.8   (slightly more exploration early on)
#   - entropy_coef kept at 0.0005 (matching proven Project 1 setting)
#   - obs_groups, clip_actions, noise_std_type all preserved from P1

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class HumanoidN1LocomotionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # ── runner ────────────────────────────────────────────────────────────
    seed               = 42
    device             = "cuda:0"
    experiment_name    = "humanoid_n1_locomotion"
    run_name           = "ppo_walk_run"
    num_steps_per_env  = 64          # same proven horizon as Project 1
    max_iterations     = 3000        # locomotion needs far more iterations
    save_interval      = 50
    obs_groups         = {"policy": ["policy"], "critic": ["critic"]}
    clip_actions       = 1.0

    # ── actor / critic networks ───────────────────────────────────────────
    policy = RslRlPpoActorCriticCfg(
        init_noise_std     = 0.8,            # slightly higher than P1 (0.5) for locomotion exploration
        noise_std_type     = "log",          # log noise decay — matches P1
        actor_hidden_dims  = [512, 256, 128],
        critic_hidden_dims = [512, 256, 128],
        activation         = "elu",
    )

    # ── PPO algorithm hyperparameters ─────────────────────────────────────
    # All values match working P1 config except entropy_coef which stays at
    # 0.0005 — locomotion does NOT need more entropy than standing did.
    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs    = 8,
        num_mini_batches       = 25,
        learning_rate          = 1e-4,
        schedule               = "adaptive",
        desired_kl             = 0.01,
        gamma                  = 0.99,
        lam                    = 0.95,
        entropy_coef           = 0.0005,
        max_grad_norm          = 1.0,
        value_loss_coef        = 1.0,
        use_clipped_value_loss = True,       # correct field name from P1
        clip_param             = 0.2,
    )
