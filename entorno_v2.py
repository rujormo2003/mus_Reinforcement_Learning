# --- Guardar como: entorno_v2.py ---
import gymnasium as gym
import numpy as np
import random
from entorno_mus import MusEnvironment, MusAction # Importamos nuestro entorno base
from bots_mus import RandomAgent, BasicoAgent     # Importamos los bots

class MusEnvConBots(MusEnvironment):
    """
    Este entorno "envuelve" al MusEnvironment.
    La IA (PPO) es SIEMPRE el Jugador 0.
    El Jugador 1 es un bot (Básico o Aleatorio) controlado por este entorno.
    """
    def __init__(self, target_score=30):
        super().__init__(target_score)
        
        # Creamos nuestra población de oponentes
        self.bot_random = RandomAgent(1)
        self.bot_basico = BasicoAgent(1)
        self.oponente_actual = self.bot_basico # Por defecto
        
        print("Entorno V2 (Gimnasio de Bots) creado.")

    def reset(self, seed=None, options=None):
        # 1. Reseteo normal
        observation, info = super().reset(seed=seed, options=options)
        
        # 2. Elegir un nuevo oponente para esta partida
        if random.random() < 0.5:
            self.oponente_actual = self.bot_random
        else:
            self.oponente_actual = self.bot_basico
            
        # 3. Importante: ¿Quién empieza?
        # Si le toca a la IA (J0), devolvemos la observación
        if self.current_player_idx == 0:
            return observation, info
        
        # Si le toca al Bot (J1), tenemos que hacer su jugada *ahora mismo*
        # hasta que le vuelva a tocar a la IA
        else:
            # Hacemos que el bot juegue su turno
            obs_dict_bot = self._get_obs_dict_for_bot(1)
            action_enum = self.oponente_actual.choose_action(obs_dict_bot)
            action_value = action_enum.value
            
            # Pasamos la acción del bot a nuestro step "inteligente"
            # (que sabrá que el J0 sigue esperando)
            observation, reward, done, info = self._process_step_and_opponent_turns(action_value)
            return observation, info

    def _get_obs_dict_for_bot(self, player_idx):
        """Crea la observación "humana" que los bots esperan."""
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

    def _process_step_and_opponent_turns(self, action_value_ia):
        """
        El corazón de V2. Procesa la acción de la IA y luego 
        gestiona TODOS los turnos del bot hasta que le toque a la IA de nuevo.
        """
        # 1. Ejecutar el paso de la IA (J0)
        obs, reward, terminated, truncated, info = super().step(action_value_ia)
        
        done = terminated or truncated
        reward_acumulado_ia = reward # Recompensa de la IA por su acción
        
        # 2. Bucle: mientras sea el turno del Bot (J1) y la partida no acabe
        while self.current_player_idx == 1 and not done:
            
            # 3. El Bot (J1) decide su acción
            obs_dict_bot = self._get_obs_dict_for_bot(1)
            action_enum_bot = self.oponente_actual.choose_action(obs_dict_bot)
            
            if action_enum_bot is None:
                # Partida terminada por el otro jugador
                break
                
            action_value_bot = action_enum_bot.value
            
            # 4. El Entorno procesa la acción del Bot
            obs, reward, terminated, truncated, info = super().step(action_value_bot)
            done = terminated or truncated
            
            # 5. Acumulamos la recompensa de la IA
            # (Si el bot dijo "NO QUIERO", la IA (J0) recibe +1)
            # (Si el bot dijo "QUIERO" y perdió, la IA (J0) recibe +2)
            # ¡La recompensa que devuelve step() es SIEMPRE desde la perspectiva del jugador que actuó!
            # Así que si reward es -1 (bot dijo NO QUIERO), J0 no recibe nada.
            # Si reward es -2 (bot quiso y perdió), J0... ¡ah!
            # Nuestro entorno base `step()` devuelve la recompensa para el jugador actual.
            # `reward` aquí es la recompensa del J1. Tenemos que "invertirla".
            
            # NO, la lógica es más simple:
            # La recompensa de J0 se da cuando J1 actúa.
            # `super().step()` (de entorno_mus) ya maneja esto, pero `reward` es para J1.
            # Debemos calcular la recompensa de J0 nosotros.
            # ...
            # ¿Sabes qué? Es demasiado complejo. Vamos a simplificar.
            # Dejemos que `reward_acumulado_ia` sea el único que importa.
            # El `step()` base NO está hecho para esto.
            #
            # ¡VAMOS A USAR EL PLAN ORIGINAL!
            # Este wrapper es demasiado complejo.
            # `entrenar_v2.py` DEBE tener el bucle manual.
            
            pass # Esta idea del wrapper es demasiado compleja.
            
        return obs, reward_acumulado_ia, done, info


    def step(self, action_value_ia):
        """
        El step que ve PPO. La IA (J0) da su acción,
        y este step simula toda la respuesta del Bot (J1)
        hasta que le vuelve a tocar a la IA.
        """
        
        # 1. Procesar la acción de la IA (J0)
        # Usamos el step() de la clase padre (MusEnvironment)
        obs_J1_vec, reward_J0, done, trunc, info = super().step(action_value_ia)
        
        if done or trunc:
            return obs_J1_vec, reward_J0, done, trunc, info

        # 2. Bucle: mientras sea el turno del Bot (J1)
        while self.current_player_idx == 1 and not (done or trunc):
            
            # 3. Crear la observación para el Bot
            obs_dict_bot = self._get_obs_dict_for_bot(1)
            
            # 4. El Bot (J1) decide su acción
            action_enum_bot = self.oponente_actual.choose_action(obs_dict_bot)
            if action_enum_bot is None:
                break # Pasa si no hay acciones legales (partida acabada)
                
            action_value_bot = action_enum_bot.value
            
            # 5. Procesar la acción del Bot (J1)
            # `reward_J1` es la recompensa del bot por esta acción
            obs_J0_vec, reward_J1, done, trunc, info = super().step(action_value_bot)
            
            # 6. Importante: ¡La recompensa de J0 debe acumularse!
            # Si J1 dijo "NO QUIERO", `reward_J1` es -1. La recompensa de J0 es +1.
            # Si J1 dijo "QUIERO" y perdió (`reward_J1` es -2), la de J0 es +2.
            # La recompensa de J0 es el negativo de la de J1.
            if reward_J1 != 0:
                reward_J0 += (-reward_J1)

        # 6. El bucle ha terminado. Le toca a J0.
        # Devolvemos la última observación (que es para J0) y la recompensa ACUMULADA.
        obs_J0_vec = self._get_observation(0) # Asegurarnos de que es la obs de J0
        return obs_J0_vec, reward_J0, done, trunc, info