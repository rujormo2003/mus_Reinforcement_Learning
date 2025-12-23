# /mus_gym/__init__.py

import gymnasium as gym

# 3. Registra el entorno en Gymnasium
gym.register(
    id="Mus-v0",
    entry_point="mus_gym.mus_env:MusEnvironment", 
    kwargs={"target_score": 30},
)

gym.register(
    id="Mus-v1",
    entry_point="mus_gym.mus_env_v2:MusEnvConBots", 
    kwargs={"target_score": 30},
)

gym.register(
    id="Mus-v2",
    entry_point="mus_gym.mus_curriculum_env:MusEnvCurriculum", 
    kwargs={"target_score": 30},
)

gym.register(
    id="Mus-v3",
    entry_point="mus_gym.mus_selfplay_env:MusEnvSelfPlay", 
    kwargs={"target_score": 30},
)