import gymnasium as gym
from . import agents  # noqa: F401

gym.register(
    id="Isaac-HumanoidN1-Stand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_n1_env_cfg:HumanoidN1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidN1PPORunnerCfg",
    },
)
