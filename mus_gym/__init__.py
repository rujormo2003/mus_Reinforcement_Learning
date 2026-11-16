"""
mus_gym: el juego de cartas Mus (1 vs 1) como entorno de Gymnasium.

Al importar el paquete se registran los entornos:

    Mus-v0  MusEnvironment    Motor base, ambos jugadores controlados desde fuera.
    Mus-v1  MusEnvConBots     Agente (J0) contra HeuristicBot (J1).
    Mus-v2  MusEnvCurriculum  Agente contra un modelo SB3 congelado (opponent_policy=...).
    Mus-v3  MusEnvSelfPlay    Self-play: rival cargado desde .zip (opponent_path=...).
"""
import gymnasium as gym

from .mus_bots import HeuristicBot, SheriffBot
from .mus_entities import Baraja, Carta, Mano
from .mus_enums import LanceState, MusAction, MusLance
from .mus_env import MusEnvironment
from .mus_opponent_env import MusEnvConBots, MusEnvCurriculum, MusEnvSelfPlay

__all__ = [
    "MusEnvironment", "MusEnvConBots", "MusEnvCurriculum", "MusEnvSelfPlay",
    "HeuristicBot", "SheriffBot", "MusAction", "MusLance", "LanceState",
    "Carta", "Baraja", "Mano",
]

_ENVS = {
    "Mus-v0": "mus_gym.mus_env:MusEnvironment",
    "Mus-v1": "mus_gym.mus_opponent_env:MusEnvConBots",
    "Mus-v2": "mus_gym.mus_opponent_env:MusEnvCurriculum",
    "Mus-v3": "mus_gym.mus_opponent_env:MusEnvSelfPlay",
}
for _env_id, _entry_point in _ENVS.items():
    if _env_id not in gym.registry:
        gym.register(id=_env_id, entry_point=_entry_point, kwargs={"target_score": 30})
