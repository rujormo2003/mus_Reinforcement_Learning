# mus_gym/mus_rules.py
from .mus_entities import Mano

def _get_winner(idx_a: int, idx_b: int, is_winner_a: bool | None, mano_player_idx: int) -> int:
    """ Determina el ganador. Si hay empate (is_winner_a is None), gana la mano. """
    if is_winner_a is None: 
        return mano_player_idx
    return idx_a if is_winner_a else idx_b

def compare_grande(mano0: Mano, mano1: Mano, mano_player_idx: int) -> int:
    vals0 = mano0.get_valores_comparacion()
    vals1 = mano1.get_valores_comparacion()
    is_winner_0 = None
    
    for i in range(4):
        if vals0[i] > vals1[i]:
            is_winner_0 = True
            break
        if vals0[i] < vals1[i]:
            is_winner_0 = False
            break
    return _get_winner(0, 1, is_winner_0, mano_player_idx)

def compare_chica(mano0: Mano, mano1: Mano, mano_player_idx: int) -> int:
    vals0 = mano0.get_valores_comparacion()
    vals1 = mano1.get_valores_comparacion()
    is_winner_0 = None
    
    for i in range(3, -1, -1):
        if vals0[i] < vals1[i]:
            is_winner_0 = True
            break
        if vals0[i] > vals1[i]:
            is_winner_0 = False
            break
    return _get_winner(0, 1, is_winner_0, mano_player_idx)

def compare_pares(mano0: Mano, mano1: Mano, mano_player_idx: int) -> int:
    info0, info1 = mano0.get_pares_info(), mano1.get_pares_info()
    rank = {'Ninguno': 0, 'Pares': 1, 'Medias': 2, 'Duples': 3}
    r0, r1 = rank[info0['tipo']], rank[info1['tipo']]
    is_winner_0 = None
    
    if r0 > r1: is_winner_0 = True
    elif r0 < r1: is_winner_0 = False
    else:
        if info0['valor_principal'] > info1['valor_principal']: is_winner_0 = True
        elif info0['valor_principal'] < info1['valor_principal']: is_winner_0 = False
        elif info0['tipo'] == 'Duples' and is_winner_0 is None:
            val_sec0, val_sec1 = info0.get('valor_secundario', 0), info1.get('valor_secundario', 0)
            if val_sec0 > val_sec1: is_winner_0 = True
            elif val_sec0 < val_sec1: is_winner_0 = False
    return _get_winner(0, 1, is_winner_0, mano_player_idx)

def compare_juego(mano0: Mano, mano1: Mano, mano_player_idx: int) -> int:
    j0, j1 = mano0.get_juego_valor(), mano1.get_juego_valor()
    tj0, tj1 = mano0.tiene_juego(), mano1.tiene_juego()
    is_winner_0 = None
    
    if tj0 and not tj1: is_winner_0 = True
    elif not tj0 and tj1: is_winner_0 = False
    elif tj0 and tj1:
        val0 = 100 if j0 == 31 else (99 if j0 == 32 else j0)
        val1 = 100 if j1 == 31 else (99 if j1 == 32 else j1)
        if val0 > val1: is_winner_0 = True
        if val0 < val1: is_winner_0 = False
    elif not tj0 and not tj1:
        if j0 > j1: is_winner_0 = True
        if j0 < j1: is_winner_0 = False
    return _get_winner(0, 1, is_winner_0, mano_player_idx)