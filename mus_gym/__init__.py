# /mus_gym/__init__.py

import gymnasium as gym

# 3. Registra el entorno en Gymnasium
gym.register(
    id="Mus-v0",
    entry_point="mus_gym.mus_env:MusEnvironment", 
    kwargs={"target_score": 30},
)