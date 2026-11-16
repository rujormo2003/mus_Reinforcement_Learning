"""
Motor del juego de Mus (versión 1 contra 1) como entorno de Gymnasium.

`MusEnvironment` contiene toda la lógica de la partida: reparto, los cuatro
lances (Grande, Chica, Pares y Juego), apuestas, recuento final y puntuación.
Los dos jugadores se controlan desde fuera llamando a `step()` por turnos
(el jugador que debe actuar está en `current_player_idx`).

Para entrenar a un único agente contra un rival fijo se usan las subclases de
`mus_opponent_env.py` (`Mus-v1`, `Mus-v2`, `Mus-v3`).
"""
from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import mus_rules
from .mus_entities import Baraja, Mano
from .mus_enums import LanceState, MusAction, MusLance
from .mus_observation import MusObservationBuilder

# Valores especiales de `pending_bets` (además de la apuesta >= 0 aceptada).
BET_RENUNCIADA = -1   # Alguien dijo "no quiero": el lance ya está cobrado.
BET_SIN_DISPUTA = -2  # Solo un jugador tiene pares/juego: no hay apuesta.


class MusEnvironment(gym.Env):
    """Mus 1 vs 1 por turnos. Ambos jugadores se controlan externamente.

    Recompensa (desde el punto de vista del jugador que acaba de actuar):
      * ``shaping_coef * (Δpuntos_propios - Δpuntos_rival) / target_score`` en cada paso.
      * ``+1`` si la partida termina y gana quien actuó, ``-1`` si pierde.

    El total acumulado de una partida queda aproximadamente en [-1.1, 1.1],
    que es la escala observada en los logs de TensorBoard del proyecto.
    """

    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, target_score: int = 30, shaping_coef: float = 0.1):
        super().__init__()

        self.target_score = target_score
        self.shaping_coef = shaping_coef
        self.baraja = Baraja()

        self.scores = {0: 0, 1: 0}
        self.mano_player_idx = 0
        self.game_over = False
        self.winner: int | None = None

        self.manos: dict[int, Mano | None] = {0: None, 1: None}
        self.historial_acciones: list[int] = []
        self.current_player_idx = 0
        self.current_lance: MusLance | None = None

        self.lance_bet = 0
        self.last_valid_bet = 0
        self.player_who_bet: int | None = None
        self.current_lance_state = LanceState.START
        self.lances_list = list(MusLance)

        self.pending_bets = {l: 0 for l in MusLance}
        self.phase_winners: dict[MusLance, int | None] = {l: None for l in MusLance}

        self.observation_builder = MusObservationBuilder
        self.observation_space_size = self.observation_builder.get_observation_space_size()

        self.action_space = spaces.Discrete(len(MusAction))
        # NOTA: los límites se mantienen tal cual para que los modelos ya
        # entrenados (.zip) sigan cargando con `RecurrentPPO.load(..., env=env)`.
        max_obs_value = max(40, self.target_score)
        self.observation_space = spaces.Box(
            low=-1.0, high=max_obs_value,
            shape=(self.observation_space_size,), dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    # API de Gymnasium
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        self.scores = {0: 0, 1: 0}
        self.mano_player_idx = random.choice([0, 1])
        self.game_over = False
        self.winner = None
        self._start_new_hand(is_first_hand=True)
        return self._get_observation(self.current_player_idx), {}

    def step(self, action_value: int):
        """Ejecuta la acción del jugador `current_player_idx`.

        Las acciones ilegales se sustituyen por una acción legal aleatoria
        (comportamiento con el que se entrenaron los modelos).
        """
        if self.game_over:
            return self._get_observation(0), 0.0, True, False, {}

        actor = self.current_player_idx
        legal = self.get_legal_actions()
        try:
            action = MusAction(int(action_value))
        except ValueError:
            action = random.choice(legal)
        if action not in legal:
            action = random.choice(legal)

        scores_before = dict(self.scores)
        self.historial_acciones.append(action.value)
        info = {
            "actor": actor,
            "action_name": action.name,
            "lance_prev": self.current_lance.name if self.current_lance else "None",
        }

        if self.current_lance_state == LanceState.START:
            self._handle_action_start(action)
        else:
            info.update(self._handle_action_response(action))

        terminated = self.game_over
        info["lance_post"] = self.current_lance.name if self.current_lance else "None"
        reward = self._compute_reward(actor, scores_before)
        if terminated:
            info["winner"] = self.winner

        return self._get_observation(self.current_player_idx), reward, terminated, False, info

    # ------------------------------------------------------------------ #
    # API pública auxiliar (usada por bots, wrappers y la interfaz)
    # ------------------------------------------------------------------ #
    def get_legal_actions(self) -> list[MusAction]:
        if self.current_lance_state == LanceState.START:
            return [MusAction.PASO, MusAction.ENVIDO, MusAction.ORDAGO]
        if self.lance_bet >= self.target_score:
            return [MusAction.QUIERO, MusAction.NO_QUIERO]
        return [MusAction.QUIERO, MusAction.NO_QUIERO, MusAction.ORDAGO]

    def get_observation(self, player_idx: int) -> np.ndarray:
        """Vector de observación (tamaño 8) desde la perspectiva de `player_idx`."""
        return self.observation_builder.build_observation(self, player_idx)

    def get_bot_observation(self, player_idx: int) -> dict:
        """Observación en forma de diccionario que esperan los bots heurísticos."""
        mano = self.manos[player_idx]
        legal = self.get_legal_actions()
        return {
            "legal_actions_enums": legal,
            "legal_actions_str": [a.name for a in legal],
            "my_hand_info": {
                "valores": mano.get_valores_comparacion(),
                "pares": mano.get_pares_info(),
                "juego": mano.get_juego_valor(),
            },
            "game_state": {
                "current_lance": self.current_lance.name if self.current_lance else "None",
                "lance_state": self.current_lance_state.name,
                "bet": self.lance_bet,
                "player_who_bet": self.player_who_bet,
                "mano_player_idx": self.mano_player_idx,
            },
            "scores": self.scores,
        }

    # Alias con los nombres antiguos (los notebooks y scripts originales los usan).
    _get_legal_actions = get_legal_actions
    _get_observation = get_observation
    _get_obs_dict_for_bot = get_bot_observation

    # ------------------------------------------------------------------ #
    # Recompensa
    # ------------------------------------------------------------------ #
    def _compute_reward(self, actor: int, scores_before: dict) -> float:
        rival = 1 - actor
        d_actor = self.scores[actor] - scores_before[actor]
        d_rival = self.scores[rival] - scores_before[rival]
        reward = self.shaping_coef * (d_actor - d_rival) / float(self.target_score)
        if self.game_over:
            reward += 1.0 if self.winner == actor else -1.0
        return float(reward)

    # ------------------------------------------------------------------ #
    # Flujo de la mano
    # ------------------------------------------------------------------ #
    def _start_new_hand(self, is_first_hand: bool = False):
        if self.game_over:
            return
        if not is_first_hand:
            self.mano_player_idx = 1 - self.mano_player_idx

        self.baraja.crear_baraja()
        self.baraja.mezclar()
        self.manos[0] = Mano(self.baraja.repartir(4))
        self.manos[1] = Mano(self.baraja.repartir(4))
        self.historial_acciones = []

        self.pending_bets = {l: 0 for l in MusLance}
        self.phase_winners = {l: None for l in MusLance}

        self.current_lance = None
        self._move_to_next_lance()

    def _set_next_lance(self, lance: MusLance):
        self.current_lance = lance
        self.current_player_idx = self.mano_player_idx
        self.lance_bet = 0
        self.last_valid_bet = 0
        self.player_who_bet = None
        self.current_lance_state = LanceState.START

    def _auto_resolve_lance(self, lance: MusLance) -> bool:
        """Pares y Juego solo se juegan si ambos jugadores tienen.

        Devuelve True si el lance se resuelve sin apuestas (se salta).
        Si solo uno tiene, se lleva 1 punto inmediato y sus puntos de
        pares/juego en el recuento final.
        """
        if lance == MusLance.PARES:
            p0_has, p1_has = self.manos[0].tiene_pares(), self.manos[1].tiene_pares()
        elif lance == MusLance.JUEGO:
            p0_has, p1_has = self.manos[0].tiene_juego(), self.manos[1].tiene_juego()
        else:
            return False

        if p0_has and p1_has:
            return False
        if p0_has:
            self.scores[0] += 1
        elif p1_has:
            self.scores[1] += 1

        self.pending_bets[lance] = BET_SIN_DISPUTA
        if p0_has != p1_has:
            self._check_game_over()
        return True

    def _move_to_next_lance(self):
        if self.game_over:
            return

        idx = self.lances_list.index(self.current_lance) if self.current_lance in self.lances_list else -1
        for next_lance in self.lances_list[idx + 1:]:
            if not self._auto_resolve_lance(next_lance):
                self._set_next_lance(next_lance)
                return
            if self.game_over:
                return

        self._resolve_showdown()
        self._start_new_hand()

    def _handle_action_start(self, action: MusAction):
        player = self.current_player_idx
        if action == MusAction.PASO:
            if player == self.mano_player_idx:
                self.current_player_idx = 1 - player
            else:
                # Ambos pasan: el lance queda "en paso" y se resuelve al final.
                self.pending_bets[self.current_lance] = 0
                self._move_to_next_lance()
        elif action in (MusAction.ENVIDO, MusAction.ORDAGO):
            self.current_lance_state = LanceState.BET_MADE
            self.lance_bet = 2 if action == MusAction.ENVIDO else self.target_score
            self.last_valid_bet = 0
            self.player_who_bet = player
            self.current_player_idx = 1 - player

    def _handle_action_response(self, action: MusAction) -> dict:
        responder = self.current_player_idx
        better = self.player_who_bet

        if action == MusAction.QUIERO:
            if self.lance_bet >= self.target_score:
                # Órdago aceptado: se enseñan las cartas y el ganador se lleva la partida.
                winner = self._resolve_lance_winner()
                self.scores[winner] = self.target_score
                self._check_game_over()
            else:
                self.pending_bets[self.current_lance] = self.lance_bet
                self._move_to_next_lance()
            return {}

        if action == MusAction.NO_QUIERO:
            # Quien apostó cobra lo último que estaba aceptado (o 1 punto si nada).
            points_won = self.last_valid_bet if self.last_valid_bet > 0 else 1
            self.scores[better] += points_won
            self.phase_winners[self.current_lance] = better
            self.pending_bets[self.current_lance] = BET_RENUNCIADA
            self._check_game_over()
            self._move_to_next_lance()
            return {"renuncia_winner": better, "renuncia_points": points_won}

        if action == MusAction.ORDAGO:
            # Órdago como respuesta a un envido.
            self.last_valid_bet = self.lance_bet
            self.lance_bet = self.target_score
            self.player_who_bet = responder
            self.current_player_idx = better
        return {}

    # ------------------------------------------------------------------ #
    # Recuento y fin de partida
    # ------------------------------------------------------------------ #
    def _resolve_lance_winner(self, lance: MusLance | None = None) -> int:
        lance = lance or self.current_lance
        m0, m1, mano = self.manos[0], self.manos[1], self.mano_player_idx
        comparators = {
            MusLance.GRANDE: mus_rules.compare_grande,
            MusLance.CHICA: mus_rules.compare_chica,
            MusLance.PARES: mus_rules.compare_pares,
            MusLance.JUEGO: mus_rules.compare_juego,
        }
        return comparators[lance](m0, m1, mano)

    def _resolve_showdown(self):
        """Recuento al final de la mano: apuestas aceptadas, pasos y bonus de pares/juego."""
        if self.game_over:
            return

        for lance in self.lances_list:
            bet = self.pending_bets[lance]

            if bet == BET_SIN_DISPUTA:
                ganador = None
                if lance == MusLance.PARES:
                    ganador = 0 if self.manos[0].tiene_pares() else (1 if self.manos[1].tiene_pares() else None)
                    if ganador is not None:
                        self.scores[ganador] += mus_rules.pares_points(self.manos[ganador])
                elif lance == MusLance.JUEGO:
                    ganador = 0 if self.manos[0].tiene_juego() else (1 if self.manos[1].tiene_juego() else None)
                    if ganador is not None:
                        self.scores[ganador] += mus_rules.juego_points(self.manos[ganador])
                continue

            if bet == BET_RENUNCIADA:
                continue

            winner = self._resolve_lance_winner(lance)
            if bet > 0:
                self.scores[winner] += bet
            elif lance in (MusLance.GRANDE, MusLance.CHICA):
                self.scores[winner] += 1  # Lance "en paso": 1 punto.

            if lance == MusLance.PARES:
                self.scores[winner] += mus_rules.pares_points(self.manos[winner])
            elif lance == MusLance.JUEGO:
                self.scores[winner] += mus_rules.juego_points(self.manos[winner])

        self._check_game_over()

    def _check_game_over(self):
        s0, s1 = self.scores[0], self.scores[1]
        if s0 >= self.target_score or s1 >= self.target_score:
            self.game_over = True
            if s0 != s1:
                self.winner = 0 if s0 > s1 else 1
            else:
                self.winner = self.mano_player_idx
