import gymnasium as gym
import numpy as np
import random
from sb3_contrib import RecurrentPPO 

from .mus_env import MusEnvironment
from .mus_enums import MusAction
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .mus_env import MusEnvironment

class MusEnvCurriculum(MusEnvironment):
    """
    Entorno dedicado al entrenamiento por Currículum (Self-Play).
    Requiere que se le pase un modelo RL congelado como oponente.
    """
    def __init__(self, target_score=30, opponent_policy=None):
        super().__init__(target_score)
        
        if opponent_policy is None:
            raise ValueError(
                "MusEnvCurriculum requiere que se pase un objeto 'opponent_policy' (modelo RL congelado) "
                "para operar. No se usa el HeuristicBot."
            )
            
        self.oponente_actual = opponent_policy
        self.is_rl_opponent = True
        print("Entorno Curriculum: Entrenando contra Modelo RL Congelado (Profesor).")

    def _oponente_choose_action(self, player_idx: int) -> MusAction:
        obs_vector = self._get_observation(player_idx)
        action_val, _ = self.oponente_actual.predict(
            obs_vector, 
            deterministic=True
        )
        return MusAction(action_val.item())

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        reward_J0 = 0.0
        while self.current_player_idx == 1 and not self.game_over:
            action_enum_bot = self._oponente_choose_action(1) 
            if action_enum_bot is None: break 
            observation, reward_J1, terminated, truncated, info = super().step(action_enum_bot.value)
            reward_J0 += (-reward_J1)
            if terminated or truncated: break
        return observation, info

    def step(self, action_value_IA: int):
        obs_J0_vec, reward_J0, done, trunc, info = super().step(action_value_IA)

        if done or trunc: return obs_J0_vec, reward_J0, done, trunc, info

        while self.current_player_idx == 1 and not (done or trunc):
            action_enum_bot = self._oponente_choose_action(1) 
            if action_enum_bot is None: break 
            obs_J0_vec, reward_J1, done, trunc, info = super().step(action_enum_bot.value)
            reward_J0 += (-reward_J1)

        return obs_J0_vec, reward_J0, done, trunc, info