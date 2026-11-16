"""
Reglas para decidir el ganador de cada lance y los puntos de pares/juego.

Todas las funciones `compare_*` devuelven el índice del jugador ganador (0 o 1).
En caso de empate gana el jugador que es "mano".
"""
from __future__ import annotations

from .mus_entities import Mano

RANKING_PARES = {"Ninguno": 0, "Pares": 1, "Medias": 2, "Duples": 3}


def _winner(is_winner_0: bool | None, mano_player_idx: int) -> int:
    if is_winner_0 is None:
        return mano_player_idx
    return 0 if is_winner_0 else 1


def _cmp(a, b) -> bool | None:
    """True si a > b, False si a < b, None si empatan."""
    if a == b:
        return None
    return a > b


def compare_grande(mano0: Mano, mano1: Mano, mano_player_idx: int) -> int:
    """Gana la carta más alta; si empatan se compara la siguiente."""
    return _winner(
        _cmp(mano0.get_valores_comparacion(), mano1.get_valores_comparacion()),
        mano_player_idx,
    )


def compare_chica(mano0: Mano, mano1: Mano, mano_player_idx: int) -> int:
    """Gana la carta más baja; si empatan se compara la siguiente más baja."""
    low0 = list(reversed(mano0.get_valores_comparacion()))
    low1 = list(reversed(mano1.get_valores_comparacion()))
    res = _cmp(low0, low1)
    return _winner(None if res is None else not res, mano_player_idx)


def compare_pares(mano0: Mano, mano1: Mano, mano_player_idx: int) -> int:
    """Duples > Medias > Pares; a igual tipo, gana la figura más alta."""
    i0, i1 = mano0.get_pares_info(), mano1.get_pares_info()
    key0 = (RANKING_PARES[i0["tipo"]], i0["valor_principal"], i0["valor_secundario"])
    key1 = (RANKING_PARES[i1["tipo"]], i1["valor_principal"], i1["valor_secundario"])
    return _winner(_cmp(key0, key1), mano_player_idx)


def _valor_juego_ordenable(puntos: int) -> int:
    """31 es el mejor juego, luego 32, y después 40, 37, 36, 35, 34, 33."""
    return {31: 100, 32: 99}.get(puntos, puntos)


def compare_juego(mano0: Mano, mano1: Mano, mano_player_idx: int) -> int:
    """Quien tiene juego gana a quien no; entre juegos, 31 > 32 > 40 > 37 > ... > 33.

    Sin juego en ninguno de los dos se compara el "punto" (suma más alta).
    """
    tj0, tj1 = mano0.tiene_juego(), mano1.tiene_juego()
    if tj0 != tj1:
        return _winner(tj0, mano_player_idx)
    j0, j1 = mano0.get_juego_valor(), mano1.get_juego_valor()
    if tj0:
        j0, j1 = _valor_juego_ordenable(j0), _valor_juego_ordenable(j1)
    return _winner(_cmp(j0, j1), mano_player_idx)


def pares_points(mano: Mano) -> int:
    """Puntos que cobra en el recuento una mano con pares (1 / 2 / 3)."""
    return {"Pares": 1, "Medias": 2, "Duples": 3}.get(mano.get_pares_info()["tipo"], 0)


def juego_points(mano: Mano) -> int:
    """Puntos que cobra en el recuento: 3 por la 31, 2 por otro juego, 1 por punto."""
    if not mano.tiene_juego():
        return 1
    return 3 if mano.get_juego_valor() == 31 else 2
