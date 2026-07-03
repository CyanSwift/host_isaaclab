import gymnasium as gym


gym.register(
    id="Unitree-G1-29dof-HoST-Ground",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ground_env_cfg:HostG129DofGroundEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.ground_env_cfg:HostG129DofGroundPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.host.agents.rsl_rl_ppo_cfg:HostG129DofPPORunnerCfg",
    },
)
