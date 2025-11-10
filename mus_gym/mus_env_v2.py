# mus_gym/entorno_v2.py
import gymnasium as gym
import numpy as np
import random

# CORRECCIÓN: Importa desde el paquete local
from .mus_env import MusEnvironment
from .mus_enums import MusAction
from .mus_bots import RandomAgent, BasicoAgent

class MusEnvConBots(MusEnvironment):
    """
    Entorno wrapper. La IA (PPO/DQN) es SIEMPRE el Jugador 0.
    El Jugador 1 es un bot (Básico o Aleatorio) controlado por este entorno.
    
    Hereda de MusEnvironment para usar su lógica de `step` y `reset` base.
    """
    def __init__(self, target_score=30):
        # Llama al __init__ de MusEnvironment
        super().__init__(target_score)
        
        # Creamos nuestra población de oponentes
        self.bot_random = RandomAgent(1)
        self.bot_basico = BasicoAgent(1)
        self.oponente_actual = self.bot_basico # Por defecto
        
        print("Entorno V2 (Gimnasio de Bots) creado.")

    # --- MÉTODO NUEVO: El "Traductor" ---
    def _get_obs_dict_for_bot(self, player_idx: int) -> dict:
        """
        Crea el diccionario de observación que esperan los bots (BasicoAgent).
        Lee el estado directamente del entorno base (self).
        """
        mano_bot = self.manos[player_idx]
        legal_actions_enums = self._get_legal_actions()

        # 1. Información de la mano (lo que el bot puede ver)
        hand_info = {
            "valores": mano_bot.get_valores_comparacion(),
            "pares": mano_bot.get_pares_info(),
            "juego": mano_bot.get_juego_valor()
        }

        # 2. Información del estado del juego
        game_state = {
            "current_lance": self.current_lance.name if self.current_lance else "None",
            "lance_state": self.current_lance_state.name if self.current_lance_state else "None",
            "bet": self.lance_bet,
            "player_who_bet": self.player_who_bet,
            "mano_player_idx": self.mano_player_idx
        }

        # 3. Acciones legales
        legal_actions_str = [a.name for a in legal_actions_enums]

        return {
            "legal_actions_enums": legal_actions_enums,
            "legal_actions_str": legal_actions_str,
            "my_hand_info": hand_info,
            "game_state": game_state,
            "scores": self.scores
        }

    # --- MÉTODO reset() SOBREESCRITO ---
    def reset(self, seed=None, options=None):
        """
        Resetea el entorno y se asegura de que si el bot (J1)
        es mano, juegue su turno antes de devolver el control.
        """
        # 1. Reseteo normal de la clase padre (MusEnvironment)
        # Esto reparte cartas y elige aleatoriamente quién es mano (J0 o J1)
        # Llama al reset de MusEnvironment, que devuelve (obs_vector, info)
        observation, info = super().reset(seed=seed, options=options)
        
        # 2. Elegir un nuevo oponente para esta partida
        if random.random() < 0.5:
            self.oponente_actual = self.bot_random
        else:
            self.oponente_actual = self.bot_basico
        
        # 3. Bucle de Turno del Bot (si J1 es mano)
        # Si el jugador actual es 1 (el Bot), debe jugar.
        reward_J0 = 0.0
        
        while self.current_player_idx == 1 and not self.game_over:
            # 3a. El Bot (J1) decide su acción
            obs_dict_bot = self._get_obs_dict_for_bot(1)
            action_enum_bot = self.oponente_actual.choose_action(obs_dict_bot)
            if action_enum_bot is None: break 
            
            # 3b. Procesar la acción del Bot (J1)
            # Llama al step de MusEnvironment
            observation, reward_J1, terminated, truncated, info = super().step(action_enum_bot.value)
            
            # 3c. Acumular recompensa (Juego de suma cero)
            reward_J0 += (-reward_J1)
            
            if terminated or truncated:
                break
        
        # 4. Devolver la observación para la IA (J0) y la info
        # 'observation' es el vector que generó el último super().step()
        return observation, info

    # --- MÉTODO step() SOBREESCRITO ---
    def step(self, action_value_IA: int):
        """
        La IA (J0) toma un paso, y luego el Bot (J1) toma
        todos los pasos que necesite hasta que sea el turno de la IA de nuevo.
        """
        # 1. Procesar la acción de la IA (J0)
        # Llama al step de MusEnvironment
        obs_J0_vec, reward_J0, done, trunc, info = super().step(action_value_IA)

        # Si la partida terminó por la acción de la IA, devolvemos el control
        if done or trunc:
            return obs_J0_vec, reward_J0, done, trunc, info

        # 2. Bucle: mientras sea el turno del Bot (J1) y la partida no acabe
        while self.current_player_idx == 1 and not (done or trunc):
            
            # 3. Crear la observación para el Bot
            obs_dict_bot = self._get_obs_dict_for_bot(1)
            
            # 4. El Bot (J1) decide su acción
            action_enum_bot = self.oponente_actual.choose_action(obs_dict_bot)
            if action_enum_bot is None:
                break 
            
            # 5. Procesar la acción del Bot (J1)
            # `reward_J1` es la recompensa del bot por esta acción
            # `obs_J0_vec` es la observación para la IA si le tocase
            obs_J0_vec, reward_J1, done, trunc, info = super().step(action_enum_bot.value)
            
            # 6. Acumular la recompensa de J0 (lógica de suma cero)
            reward_J0 += (-reward_J1)

        # 7. El bucle ha terminado. Es el turno de la IA (J0) o la partida ha acabado.
        # Devolvemos la última observación de J0, la recompensa ACUMULADA, y el estado.
        return obs_J0_vec, reward_J0, done, trunc, info