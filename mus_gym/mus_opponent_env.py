"""
Entornos de un solo agente: el agente RL es siempre el Jugador 0 y el
Jugador 1 lo controla un rival interno (bot heurístico o modelo congelado).

Tras cada acción del agente, el rival juega automáticamente todos sus turnos
hasta que vuelve a tocarle al agente (o termina la partida). La recompensa
que recibe el agente es la suya menos la que obtiene el rival en sus turnos
(juego de suma cero).
"""
from __future__ import annotations

from .mus_bots import HeuristicBot
from .mus_enums import MusAction
from .mus_env import MusEnvironment

AGENT, OPPONENT = 0, 1


class MusOpponentEnv(MusEnvironment):
    """Clase base: subclases implementan `_opponent_action()`."""

    def _opponent_action(self) -> MusAction | None:
        raise NotImplementedError

    def _play_opponent_turns(self, obs, reward: float, info: dict):
        terminated = self.game_over
        while self.current_player_idx == OPPONENT and not terminated:
            action = self._opponent_action()
            if action is None:
                break
            obs, r_opp, terminated, _, info = super().step(action.value)
            reward -= r_opp
        return obs, reward, terminated, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        obs, _, _, info = self._play_opponent_turns(obs, 0.0, info)
        return obs, info

    def step(self, action_value: int):
        obs, reward, terminated, truncated, info = super().step(action_value)
        if not terminated:
            obs, reward, terminated, info = self._play_opponent_turns(obs, reward, info)
        return obs, reward, terminated, truncated, info


class MusEnvConBots(MusOpponentEnv):
    """`Mus-v1`: el rival es un bot heurístico (por defecto `HeuristicBot`)."""

    def __init__(self, target_score: int = 30, bot: HeuristicBot | None = None, **kwargs):
        super().__init__(target_score, **kwargs)
        self.bot_oponente = bot or HeuristicBot(player_id=OPPONENT, prob_farol=0.15,
                                                prob_error=0.05, prob_hero_call=0.10)

    def _opponent_action(self) -> MusAction | None:
        return self.bot_oponente.choose_action(self.get_bot_observation(OPPONENT))


class MusEnvCurriculum(MusOpponentEnv):
    """`Mus-v2`: el rival es una política de SB3 congelada (objeto con `.predict`).

    Nota: el rival predice sin estado LSTM (cada decisión parte de un estado
    cero), que es como se entrenaron los modelos publicados.
    """

    def __init__(self, target_score: int = 30, opponent_policy=None, **kwargs):
        super().__init__(target_score, **kwargs)
        if opponent_policy is None:
            raise ValueError("MusEnvCurriculum necesita 'opponent_policy' (un modelo SB3 cargado).")
        self.oponente_actual = opponent_policy

    def _opponent_action(self) -> MusAction:
        obs = self.get_observation(OPPONENT)
        action, _ = self.oponente_actual.predict(obs, deterministic=True)
        return MusAction(int(action))


class MusEnvSelfPlay(MusEnvCurriculum):
    """`Mus-v3`: self-play. Carga el rival desde un .zip y permite cambiarlo en caliente."""

    def __init__(self, target_score: int = 30, opponent_path: str | None = None, **kwargs):
        if opponent_path is None:
            raise ValueError("MusEnvSelfPlay necesita 'opponent_path' (ruta a un .zip de RecurrentPPO).")
        super().__init__(target_score, opponent_policy=self._load_policy(opponent_path), **kwargs)

    @staticmethod
    def _load_policy(path: str):
        from sb3_contrib import RecurrentPPO  # Import diferido: solo hace falta aquí.
        return RecurrentPPO.load(path, device="cpu")

    def update_opponent_model(self, new_model_path: str):
        """Llamado desde los callbacks de entrenamiento (vía `env_method`)."""
        self.oponente_actual = self._load_policy(new_model_path)
