import gymnasium as gym
import numpy as np
import random

from .mus_env import MusEnvironment
from .mus_enums import MusAction
from .mus_bots import HeuristicBot

class MusEnvConBots(MusEnvironment):
    """
    Entorno wrapper. La IA (PPO/DQN) es SIEMPRE el Jugador 0.
    El Jugador 1 es ahora SIEMPRE el HeuristicBot.
    
    Hereda de MusEnvironment para usar su lógica de `step` y `reset` base.
    """
    def __init__(self, target_score=30):
        super().__init__(target_score)
        self.bot_oponente = HeuristicBot(player_id=1, prob_farol=0.15, prob_error=0.05, prob_hero_call=0.10)
        
        self.oponente_actual = self.bot_oponente 
        print("Entorno V2 creado: Entrenando EXCLUSIVAMENTE contra HeuristicBot.")

    def _get_obs_dict_for_bot(self, player_idx: int) -> dict:
        """
        Crea el diccionario de observación que espera el HeuristicBot.
        """
        mano_bot = self.manos[player_idx]
        legal_actions_enums = self._get_legal_actions()

        hand_info = {
            "valores": mano_bot.get_valores_comparacion(),
            "pares": mano_bot.get_pares_info(),
            "juego": mano_bot.get_juego_valor()
        }

        game_state = {
            "current_lance": self.current_lance.name if self.current_lance else "None",
            "lance_state": self.current_lance_state.name if self.current_lance_state else "None",
            "bet": self.lance_bet,
            "player_who_bet": self.player_who_bet,
            "mano_player_idx": self.mano_player_idx
        }

        legal_actions_str = [a.name for a in legal_actions_enums]

        return {
            "legal_actions_enums": legal_actions_enums,
            "legal_actions_str": legal_actions_str,
            "my_hand_info": hand_info,
            "game_state": game_state,
            "scores": self.scores
        }

    def reset(self, seed=None, options=None):
        """
        Resetea el entorno y hace que el Bot juegue si es mano.
        """
        observation, info = super().reset(seed=seed, options=options)
        self.oponente_actual = self.bot_oponente
        reward_J0 = 0.0
        
        while self.current_player_idx == 1 and not self.game_over:
            obs_dict_bot = self._get_obs_dict_for_bot(1)
            action_enum_bot = self.oponente_actual.choose_action(obs_dict_bot)
            if action_enum_bot is None: break 
            
            observation, reward_J1, terminated, truncated, info = super().step(action_enum_bot.value)
            
            reward_J0 += (-reward_J1)
            
            if terminated or truncated:
                break
        
        return observation, info

    def step(self, action_value_IA: int):
        """
        La IA (J0) mueve, luego el Bot (J1) responde hasta que vuelve a tocarle a la IA.
        """
        obs_J0_vec, reward_J0, done, trunc, info = super().step(action_value_IA)

        if done or trunc:
            return obs_J0_vec, reward_J0, done, trunc, info

        while self.current_player_idx == 1 and not (done or trunc):
            
            obs_dict_bot = self._get_obs_dict_for_bot(1)
            action_enum_bot = self.oponente_actual.choose_action(obs_dict_bot)
            
            if action_enum_bot is None:
                break 
            
            obs_J0_vec, reward_J1, done, trunc, info = super().step(action_enum_bot.value)
            
            reward_J0 += (-reward_J1)

        return obs_J0_vec, reward_J0, done, trunc, info