import gymnasium as gym
import numpy as np
import random
from entorno_mus import MusEnvironment, MusAction # Importamos nuestro entorno base
from bots_mus import RandomAgent, BasicoAgent     # Importamos los bots

class MusEnvConBots(MusEnvironment):
    """
    Entorno wrapper. La IA (PPO) es SIEMPRE el Jugador 0.
    El Jugador 1 es un bot (Básico o Aleatorio) controlado por este entorno.
    """
    def __init__(self, target_score=30):
        super().__init__(target_score)
        
        # Creamos nuestra población de oponentes
        self.bot_random = RandomAgent(1)
        self.bot_basico = BasicoAgent(1)
        self.oponente_actual = self.bot_basico # Por defecto
        
        print("Entorno V2 (Gimnasio de Bots) creado.")

    # --- MÉTODO reset() CORREGIDO Y LIMPIADO ---
    def reset(self, seed=None, options=None):
        # 1. Reseteo normal de la clase padre (MusEnvironment)
        # Esto reparte cartas y elige aleatoriamente quién es mano (J0 o J1)
        observation, info = super().reset(seed=seed, options=options)
        
        # 2. Elegir un nuevo oponente para esta partida
        if random.random() < 0.5:
            self.oponente_actual = self.bot_random
            # print("Oponente: RandomAgent") # Descomentar para debug
        else:
            self.oponente_actual = self.bot_basico
            # print("Oponente: BasicoAgent") # Descomentar para debug
            
        # 3. Manejar el caso de que el Bot (J1) empiece
        
        # Si le toca a la IA (J0), devolvemos la observación y terminamos.
        if self.current_player_idx == 0:
            return observation, info
        
        # Si le toca al Bot (J1), simulamos sus turnos hasta que le toque a la IA
        else:
            done = False
            trunc = False
            
            # Bucle de turnos del Bot (replica la lógica de step())
            while self.current_player_idx == 1 and not (done or trunc):
                obs_dict_bot = self._get_obs_dict_for_bot(1)
                action_enum_bot = self.oponente_actual.choose_action(obs_dict_bot)
                
                if action_enum_bot is None:
                    break # Partida terminada
                
                action_value_bot = action_enum_bot.value
                
                # Procesamos la acción del Bot.
                # obs_J0_vec es la obs para la IA, reward_J1 es la recompensa del bot
                obs_J0_vec, reward_J1, done, trunc, info = super().step(action_value_bot)
                
                # En reset, no hay recompensa acumulada que devolver.
                # Solo necesitamos que el bucle avance hasta que le toque a J0.
            
            # El bucle ha terminado. Le toca a J0 o la partida ha acabado.
            # Nos aseguramos de devolver la observación MÁS RECIENTE para J0.
            final_obs_J0 = self._get_observation(0)
            return final_obs_J0, info

    def _get_obs_dict_for_bot(self, player_idx):
        """Crea la observación "humana" que los bots esperan."""
        # Esta función estaba perfecta, sin cambios.
        return {
            "my_hand_info": {
                "valores": self.manos[player_idx].get_valores_comparacion(),
                "pares": self.manos[player_idx].get_pares_info(),
                "juego": self.manos[player_idx].get_juego_valor(),
                "tiene_juego": self.manos[player_idx].tiene_juego()
            },
            "game_state": {
                "current_lance": self.current_lance,
                "lance_state": self.lance_state,
            },
            "legal_actions_enums": self._get_legal_actions()
        }

    # --- MÉTODO ABANDONADO _process_step... HA SIDO ELIMINADO ---

    def step(self, action_value_ia):
        """
        El step que ve PPO. La IA (J0) da su acción,
        y este step simula toda la respuesta del Bot (J1)
        hasta que le vuelve a tocar a la IA.
        """
        
        # 1. Procesar la acción de la IA (J0)
        # obs_J1_vec es la obs del bot, reward_J0 es la recompensa de la IA
        obs_J1_vec, reward_J0, done, trunc, info = super().step(action_value_ia)
        
        if done or trunc:
            # Si la acción de la IA terminó la partida (ej. un Órdago ilegal)
            return obs_J1_vec, reward_J0, done, trunc, info

        # 2. Bucle: mientras sea el turno del Bot (J1) y la partida no acabe
        while self.current_player_idx == 1 and not (done or trunc):
            
            # 3. Crear la observación para el Bot
            obs_dict_bot = self._get_obs_dict_for_bot(1)
            
            # 4. El Bot (J1) decide su acción
            action_enum_bot = self.oponente_actual.choose_action(obs_dict_bot)
            if action_enum_bot is None:
                break 
            
            action_value_bot = action_enum_bot.value
            
            # 5. Procesar la acción del Bot (J1)
            # `reward_J1` es la recompensa del bot por esta acción
            obs_J0_vec, reward_J1, done, trunc, info = super().step(action_value_bot)
            
            # 6. Acumular la recompensa de J0 (lógica de suma cero)
            # Asumimos que entorno_mus.py tiene las recompensas normalizadas.
            # Si J1 dijo "NO QUIERO", reward_J1 es -0.033. J0 debe ganar +0.033.
            if reward_J1 != 0:
                reward_J0 += (-reward_J1)

        # 7. El bucle ha terminado. Le toca a J0.
        # Devolvemos la última observación (que es para J0) y la recompensa ACUMULADA.
        obs_J0_vec = self._get_observation(0) # Asegurarnos de que es la obs de J0
        return obs_J0_vec, reward_J0, done, trunc, info