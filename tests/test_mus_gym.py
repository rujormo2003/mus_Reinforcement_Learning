"""Tests de reglas y entornos. Ejecutar con: python -m pytest"""
import random

import numpy as np
import pytest

from mus_gym import HeuristicBot, MusAction, MusEnvConBots, MusEnvironment
from mus_gym import mus_rules
from mus_gym.mus_entities import Carta, Mano
from mus_gym.mus_observation import MusObservationBuilder


def mano(*rangos, palo="Oros"):
    return Mano([Carta(palo, r) for r in rangos])


# --------------------------------------------------------------------- #
# Entidades
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("rangos, tipo", [
    (("Rey", "Caballo", "5", "As"), "Ninguno"),
    (("Rey", "3", "5", "As"), "Pares"),          # el 3 cuenta como rey
    (("As", "2", "As", "5"), "Medias"),          # el 2 cuenta como as
    (("Rey", "Rey", "Sota", "Sota"), "Duples"),
    (("4", "4", "4", "4"), "Duples"),
])
def test_tipo_de_pares(rangos, tipo):
    assert mano(*rangos).get_pares_info()["tipo"] == tipo


def test_valor_de_juego():
    m = mano("Rey", "Caballo", "Sota", "As")
    assert m.get_juego_valor() == 31 and m.tiene_juego()
    assert not mano("4", "5", "6", "7").tiene_juego()


# --------------------------------------------------------------------- #
# Reglas de cada lance
# --------------------------------------------------------------------- #
def test_grande_y_chica():
    alta, baja = mano("Rey", "Rey", "5", "4"), mano("Caballo", "As", "As", "4")
    assert mus_rules.compare_grande(alta, baja, mano_player_idx=1) == 0
    assert mus_rules.compare_chica(alta, baja, mano_player_idx=0) == 1


def test_empate_gana_la_mano():
    a, b = mano("Rey", "5", "4", "As"), mano("Rey", "5", "4", "As", palo="Copas")
    assert mus_rules.compare_grande(a, b, mano_player_idx=1) == 1
    assert mus_rules.compare_chica(a, b, mano_player_idx=0) == 0


def test_pares():
    duples, medias = mano("4", "4", "5", "5"), mano("Rey", "Rey", "Rey", "As")
    assert mus_rules.compare_pares(duples, medias, 1) == 0
    assert mus_rules.compare_pares(mano("Rey", "Rey", "4", "5"), mano("Sota", "Sota", "4", "5"), 1) == 0
    assert mus_rules.compare_pares(mano("Rey", "Rey", "4", "4"), mano("Rey", "Rey", "5", "5"), 0) == 1


def test_juego_31_gana_a_40():
    j31, j40 = mano("Rey", "Caballo", "Sota", "As"), mano("Rey", "Rey", "Rey", "Rey")
    assert mus_rules.compare_juego(j31, j40, 1) == 0
    assert mus_rules.compare_juego(mano("Rey", "Rey", "6", "6"), j40, 1) == 0  # 32 > 40


def test_puntos_recuento():
    assert mus_rules.pares_points(mano("Rey", "Rey", "4", "4")) == 3
    assert mus_rules.juego_points(mano("Rey", "Caballo", "Sota", "As")) == 3
    assert mus_rules.juego_points(mano("Rey", "Rey", "Rey", "Rey")) == 2


# --------------------------------------------------------------------- #
# Entorno
# --------------------------------------------------------------------- #
def jugar_partida(env, policy):
    obs, _ = env.reset(seed=random.randint(0, 10_000))
    total, done, steps = 0.0, False, 0
    while not done:
        assert env.observation_space.shape == obs.shape
        obs, r, done, _, _ = env.step(policy(env))
        total += r
        steps += 1
        assert steps < 2000
    return total


def test_partidas_aleatorias_terminan_y_suma_cero():
    random.seed(0)
    env = MusEnvironment()
    for _ in range(50):
        env.reset(seed=random.randint(0, 10_000))
        rewards = {0: 0.0, 1: 0.0}
        done = False
        while not done:
            actor = env.current_player_idx
            _, r, done, _, _ = env.step(random.choice(env.get_legal_actions()).value)
            rewards[actor] += r          # suma cero: lo que gana uno lo pierde el otro
            rewards[1 - actor] -= r
        assert env.winner in (0, 1)
        assert max(env.scores.values()) >= 30
        assert rewards[env.winner] > 0 > rewards[1 - env.winner]


def test_recompensa_terminal_en_mus_v1():
    random.seed(1)
    env = MusEnvConBots()
    for _ in range(30):
        total = jugar_partida(env, lambda e: random.choice(e.get_legal_actions()).value)
        assert (total > 0) == (env.winner == 0)
        assert -1.2 <= total <= 1.2


def test_ordago_aceptado_termina_la_partida():
    env = MusEnvironment()
    env.reset(seed=3)
    env.step(MusAction.ORDAGO.value)
    env.step(MusAction.QUIERO.value)
    assert env.game_over and env.scores[env.winner] == 30


def test_accion_ilegal_se_sustituye():
    env = MusEnvironment()
    env.reset(seed=4)
    _, _, _, _, info = env.step(MusAction.QUIERO.value)  # ilegal sin apuesta previa
    assert info["action_name"] in {"PASO", "ENVIDO", "ORDAGO"}


def test_observacion_en_rango():
    env = MusEnvironment()
    env.reset(seed=5)
    obs = MusObservationBuilder.build_observation(env, 0)
    assert obs.dtype == np.float32 and obs.shape == (8,)
    assert np.all(obs[:7] >= -1) and np.all(obs[:7] <= 1)


def test_heuristic_bot_siempre_legal():
    random.seed(2)
    env, bot = MusEnvironment(), HeuristicBot()
    for _ in range(20):
        env.reset()
        while not env.game_over:
            act = bot.choose_action(env.get_bot_observation(env.current_player_idx))
            assert act in env.get_legal_actions()
            env.step(act.value)
