# mus_gym/mus_observation.py
import numpy as np
from .mus_enums import MusLance, LanceState
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .mus_env import MusEnvironment

class MusObservationBuilder:
    """ 
    Constructor de observaciones simplificado (Feature Engineering).
    
    Vector de tamaño 8:
    [Reyes, Ases, Pares, Juego, LanceActual, EstadoApuesta, SoyMano, ScoreDiff]
    """
    
    OBS_SIZE = 8
    
    @classmethod
    def get_observation_space_size(cls) -> int:
        return cls.OBS_SIZE

    @classmethod
    def build_observation(cls, env: 'MusEnvironment', player_idx: int) -> np.ndarray:
        obs = np.zeros(cls.OBS_SIZE, dtype=np.float32)
        mano = env.manos[player_idx]
        vals = mano.get_valores_comparacion()
        
        # --- 1. Reyes (Normalizado 0-1) ---
        # Contamos cartas con valor 10 (Reyes y 3s en configuración Mus)
        num_reyes = sum(1 for v in vals if v == 10)
        obs[0] = num_reyes / 4.0

        # --- 2. Ases (Normalizado 0-1) ---
        # Contamos cartas con valor 1 (Ases y 2s)
        num_ases = sum(1 for v in vals if v == 1)
        obs[1] = num_ases / 4.0

        # --- 3. Escala de Pares ---
        info_pares = mano.get_pares_info()
        tipo_pares = info_pares['tipo']
        if tipo_pares == 'Ninguno': obs[2] = 0.0
        elif tipo_pares == 'Pares': obs[2] = 0.33
        elif tipo_pares == 'Medias': obs[2] = 0.66
        elif tipo_pares == 'Duples': obs[2] = 1.0

        # --- 4. Escala de Juego (Ranking Correcto) ---
        # Solo valoramos si tiene juego (31+). Si no, es 0.0 (Punto omitido).
        obs[3] = cls._calcular_fuerza_juego(mano.get_juego_valor(), mano.tiene_juego())

        # --- 5. Lance Actual ---
        if env.current_lance == MusLance.GRANDE: obs[4] = 0.0
        elif env.current_lance == MusLance.CHICA: obs[4] = 0.33
        elif env.current_lance == MusLance.PARES: obs[4] = 0.66
        elif env.current_lance == MusLance.JUEGO: obs[4] = 1.0
        else: obs[4] = -1.0

        # --- 6. Estado Apuesta ---
        if env.current_lance_state == LanceState.START:
            obs[5] = 0.0
        else:
            # 1.0 si es órdago (o apuesta supera target), 0.5 si es envite
            obs[5] = 1.0 if env.lance_bet >= env.target_score else 0.5

        # --- 7. Soy Mano ---
        obs[6] = 1.0 if player_idx == env.mano_player_idx else 0.0

        # --- 8. Diferencia de Puntos ---
        diff = env.scores[player_idx] - env.scores[1 - player_idx]
        obs[7] = diff / float(env.target_score)

        return obs

    @staticmethod
    def _calcular_fuerza_juego(puntos: int, tiene_juego: bool) -> float:
        """ 
        Asigna valor según la jerarquía estricta del Mus.
        Como omitimos el Punto, si no hay juego devolvemos 0.0.
        """
        if not tiene_juego:
            return 0.0
            
        # Ranking de PEOR (33) a MEJOR (31)
        # Nota: 38 y 39 no existen en el Mus.
        ranking_juego = [33, 34, 35, 36, 37, 40, 32, 31]
        
        if puntos in ranking_juego:
            idx = ranking_juego.index(puntos)
            # Mapeamos linealmente de ~0.1 a 1.0
            # idx 0 (33) -> 0.125
            # idx 7 (31) -> 1.0
            return (idx + 1) / len(ranking_juego)
            
        return 0.0 # Fallback