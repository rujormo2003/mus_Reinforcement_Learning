# mus_gym/mus_observation.py
import numpy as np
from .mus_enums import MusLance, LanceState
# Usamos un type hint forward para evitar la dependencia circular
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .mus_env import MusEnvironment


class MusObservationBuilder:
    """ Clase dedicada a construir el vector de observación. """
    
    # Definimos los tamaños aquí
    OBS_MANO_SIZE = 11
    OBS_JUEGO_SIZE = 11
    OBS_HIST_SIZE = 6
    
    @classmethod
    def get_observation_space_size(cls) -> int:
        """ Devuelve el tamaño total del vector de observación. """
        return cls.OBS_MANO_SIZE + cls.OBS_JUEGO_SIZE + cls.OBS_HIST_SIZE

    @classmethod
    def build_observation(cls, env: 'MusEnvironment', player_idx: int) -> np.ndarray:
        """ Construye el vector de observación para un jugador dado. """
        
        obs = np.zeros(cls.get_observation_space_size(), dtype=np.float32)
        idx = 0
        lances_list = list(MusLance)
        lance_states_list = list(LanceState)

        # --- 1. Info de Mano (11) ---
        mano = env.manos[player_idx]
        obs[idx : idx + 4] = mano.get_valores_comparacion()
        idx += 4
        
        pares_info = mano.get_pares_info()
        if pares_info['tipo'] == 'Pares':   obs[idx] = 1.0
        elif pares_info['tipo'] == 'Medias': obs[idx + 1] = 1.0
        elif pares_info['tipo'] == 'Duples': obs[idx + 2] = 1.0
        idx += 3
        
        obs[idx] = pares_info['valor_principal']
        obs[idx + 1] = pares_info.get('valor_secundario', 0)
        idx += 2
        
        obs[idx] = 1.0 if mano.tiene_juego() else 0.0
        obs[idx + 1] = mano.get_juego_valor()
        idx += 2
        assert idx == cls.OBS_MANO_SIZE

        # --- 2. Info de Juego (11) ---
        obs[idx] = env.scores[player_idx]
        obs[idx + 1] = env.scores[1 - player_idx] 
        idx += 2
        obs[idx] = 1.0 if player_idx == env.mano_player_idx else 0.0
        idx += 1
        
        if env.current_lance in lances_list:
            obs[idx + lances_list.index(env.current_lance)] = 1.0
        idx += len(lances_list)
        
        if env.current_lance_state in lance_states_list:
            obs[idx + lance_states_list.index(env.current_lance_state)] = 1.0
        idx += len(lance_states_list)

        obs[idx] = env.lance_bet
        if env.player_who_bet is not None:
            obs[idx + 1] = 1.0 if env.player_who_bet == player_idx else -1.0
        idx += 2
        assert idx == cls.OBS_MANO_SIZE + cls.OBS_JUEGO_SIZE

        # --- 3. Historial (6) ---
        hist_len = len(env.historial_acciones)
        start_idx = idx
        end_idx = start_idx + cls.OBS_HIST_SIZE
        
        if hist_len > 0:
            num_a_tomar = min(hist_len, cls.OBS_HIST_SIZE)
            hist_slice = env.historial_acciones[-num_a_tomar:]
            obs[end_idx - num_a_tomar : end_idx] = hist_slice
            if num_a_tomar < cls.OBS_HIST_SIZE:
                obs[start_idx : end_idx - num_a_tomar] = -1.0
        else:
            obs[start_idx : end_idx] = -1.0
        idx += cls.OBS_HIST_SIZE
        
        assert idx == cls.get_observation_space_size()
        return obs