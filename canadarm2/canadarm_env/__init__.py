from gymnasium.envs.registration import register

register(
    id="Canadarm-v0",
    entry_point="canadarm_env.env_canadarm:CanadarmEnv",
)
