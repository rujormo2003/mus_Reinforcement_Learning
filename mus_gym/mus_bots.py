"""Bots con reglas fijas, usados como rivales de entrenamiento y evaluación."""
from __future__ import annotations

import random

from .mus_enums import MusAction
from .mus_observation import RANKING_JUEGO


class HeuristicBot:
    """Bot heurístico basado en umbrales de "fuerza" de la mano (0..1).

    Args:
        prob_farol: probabilidad de envidar con mano mala (farol).
        prob_error: probabilidad de jugar una acción legal al azar.
        prob_hero_call: probabilidad de aceptar un órdago con mano media.
    """

    def __init__(self, player_id: int = 1, prob_farol: float = 0.15,
                 prob_error: float = 0.05, prob_hero_call: float = 0.10):
        self.player_id = player_id
        self.prob_farol = prob_farol
        self.prob_error = prob_error
        self.prob_hero_call = prob_hero_call

    def choose_action(self, observation: dict) -> MusAction | None:
        legal = observation["legal_actions_enums"]
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]
        if random.random() < self.prob_error:
            return random.choice(legal)

        lance = observation["game_state"]["current_lance"]
        state = observation["game_state"]["lance_state"]
        fuerza = self._calcular_fuerza(lance, observation["my_hand_info"])

        if state == "START":
            return self._abrir(lance, fuerza, legal)
        return self._responder(lance, fuerza, legal, observation["game_state"]["bet"])

    # ------------------------------------------------------------------ #
    def _abrir(self, lance: str, fuerza: float, legal: list[MusAction]) -> MusAction:
        """Decisión cuando nadie ha apostado todavía."""
        envido = MusAction.ENVIDO if MusAction.ENVIDO in legal else MusAction.PASO
        umbral_envite = 0.45 if lance in ("GRANDE", "CHICA") else 0.60

        if fuerza > 0.90:
            if MusAction.ORDAGO in legal and random.random() < 0.5:
                return MusAction.ORDAGO
            return envido
        if fuerza > umbral_envite:
            return envido
        if fuerza < 0.4 and random.random() < self.prob_farol:
            return envido
        return MusAction.PASO

    def _responder(self, lance: str, fuerza: float, legal: list[MusAction], bet: int) -> MusAction:
        """Decisión ante una apuesta del rival."""
        if bet >= 30:  # Órdago
            if fuerza >= 0.95:
                return MusAction.QUIERO
            if fuerza >= 0.5 and random.random() < self.prob_hero_call:
                return MusAction.QUIERO
            return MusAction.NO_QUIERO

        if fuerza > 0.75:
            if MusAction.ORDAGO in legal and random.random() < 0.1:
                return MusAction.ORDAGO
            return MusAction.QUIERO

        umbral_quiero = 0.45 if lance in ("GRANDE", "CHICA") else 0.5
        return MusAction.QUIERO if fuerza > umbral_quiero else MusAction.NO_QUIERO

    @staticmethod
    def _calcular_fuerza(lance: str, hand: dict) -> float:
        """Estimación 0..1 de lo buena que es la mano para el lance actual."""
        vals = hand["valores"]
        if lance == "GRANDE":
            return (sum(vals) - 4) / 36.0
        if lance == "CHICA":
            return 1.0 - (sum(vals) - 4) / 36.0
        if lance == "PARES":
            info = hand["pares"]
            base = {"Pares": 0.4, "Medias": 0.7, "Duples": 0.9}.get(info["tipo"])
            if base is None:
                return 0.0
            return min(1.0, base + info["valor_principal"] / 100.0)
        if lance == "JUEGO":
            pts = hand["juego"]
            if pts in RANKING_JUEGO:
                return (RANKING_JUEGO.index(pts) + 1) / len(RANKING_JUEGO)
        return 0.0


class SheriffBot(HeuristicBot):
    """Variante que acepta los órdagos con manos medias.

    Se usó en Colab (fase 1 del currículum) para castigar los órdagos
    indiscriminados de la IA. `quiero_threshold=0.25` corresponde al
    "SuperSheriffBot" de la fase de corrección.
    """

    def __init__(self, player_id: int = 1, quiero_threshold: float = 0.40,
                 prob_farol: float = 0.05, prob_error: float = 0.02):
        super().__init__(player_id=player_id, prob_farol=prob_farol, prob_error=prob_error)
        self.quiero_threshold = quiero_threshold

    def choose_action(self, observation: dict) -> MusAction | None:
        legal = observation["legal_actions_enums"]
        if not legal:
            return None
        state = observation["game_state"]
        if state["bet"] >= 30:
            fuerza = self._calcular_fuerza(state["current_lance"], observation["my_hand_info"])
            if fuerza > self.quiero_threshold and MusAction.QUIERO in legal:
                return MusAction.QUIERO
            if MusAction.NO_QUIERO in legal:
                return MusAction.NO_QUIERO
        return super().choose_action(observation)
