"""
Construcción del vector de observación que ve la red neuronal.

IMPORTANTE: los modelos entrenados dependen de esta codificación exacta.
Si se cambia, hay que reentrenar (y actualizar la versión JavaScript en
`web/js/mus-engine.js`, que la replica para la demo web).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .mus_enums import LanceState, MusLance

if TYPE_CHECKING:
    from .mus_env import MusEnvironment

# Orden de "calidad" del juego: la 31 es la mejor, la 33 la peor.
RANKING_JUEGO = [33, 34, 35, 36, 37, 40, 32, 31]

CODIGO_PARES = {"Ninguno": 0.0, "Pares": 0.33, "Medias": 0.66, "Duples": 1.0}
CODIGO_LANCE = {MusLance.GRANDE: 0.0, MusLance.CHICA: 0.33, MusLance.PARES: 0.66, MusLance.JUEGO: 1.0}


class MusObservationBuilder:
    """Vector de 8 features:

    | idx | feature        | valores                                          |
    |-----|----------------|--------------------------------------------------|
    | 0   | reyes          | nº de reyes (incluye treses) / 4                 |
    | 1   | ases           | nº de ases (incluye doses) / 4                   |
    | 2   | pares          | 0 nada · 0.33 pares · 0.66 medias · 1 duples     |
    | 3   | juego          | posición en RANKING_JUEGO / 8 (0 sin juego)      |
    | 4   | lance actual   | 0 grande · 0.33 chica · 0.66 pares · 1 juego     |
    | 5   | estado apuesta | 0 sin apuesta · 0.5 envido · 1 órdago            |
    | 6   | soy mano       | 1 / 0                                            |
    | 7   | dif. marcador  | (mis puntos − rival) / target_score              |
    """

    OBS_SIZE = 8

    @classmethod
    def get_observation_space_size(cls) -> int:
        return cls.OBS_SIZE

    @classmethod
    def build_observation(cls, env: "MusEnvironment", player_idx: int) -> np.ndarray:
        obs = np.zeros(cls.OBS_SIZE, dtype=np.float32)
        mano = env.manos[player_idx]
        vals = mano.get_valores_comparacion()

        obs[0] = sum(1 for v in vals if v == 10) / 4.0
        obs[1] = sum(1 for v in vals if v == 1) / 4.0
        obs[2] = CODIGO_PARES[mano.get_pares_info()["tipo"]]
        obs[3] = cls._calcular_fuerza_juego(mano.get_juego_valor(), mano.tiene_juego())
        obs[4] = CODIGO_LANCE.get(env.current_lance, -1.0)

        if env.current_lance_state == LanceState.START:
            obs[5] = 0.0
        else:
            obs[5] = 1.0 if env.lance_bet >= env.target_score else 0.5

        obs[6] = 1.0 if player_idx == env.mano_player_idx else 0.0
        obs[7] = (env.scores[player_idx] - env.scores[1 - player_idx]) / float(env.target_score)
        return obs

    @staticmethod
    def _calcular_fuerza_juego(puntos: int, tiene_juego: bool) -> float:
        """Fuerza del juego según la jerarquía del Mus. Sin juego (punto) vale 0."""
        if not tiene_juego or puntos not in RANKING_JUEGO:
            return 0.0
        return (RANKING_JUEGO.index(puntos) + 1) / len(RANKING_JUEGO)
