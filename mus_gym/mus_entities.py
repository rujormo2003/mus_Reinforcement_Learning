from __future__ import annotations
import random
from collections import Counter
from .mus_config import PALOS, RANGOS_REALES, VALORES_JUEGO, VALORES_COMPARACION

class Carta:
    """ Representa una única carta de la baraja española. """
    def __init__(self, palo: str, rango: str):
        self.palo = palo
        self.rango = rango
        self.valor_juego = VALORES_JUEGO[rango]
        self.valor_comparacion = VALORES_COMPARACION[rango]
    def __repr__(self): 
        return f"{self.rango} de {self.palo}"

class Baraja:
    """ Representa una baraja de 40 cartas que se puede mezclar y repartir. """
    def __init__(self):
        self.cartas = []
        self.crear_baraja()

    def crear_baraja(self):
        self.cartas = [Carta(palo, rango) for palo in PALOS for rango in RANGOS_REALES]

    def mezclar(self): 
        random.shuffle(self.cartas)

    def repartir(self, num_cartas: int) -> list[Carta]:
        if len(self.cartas) < num_cartas:
            self.crear_baraja()
            self.mezclar()
        return [self.cartas.pop() for _ in range(num_cartas)]

class Mano:
    """ 
    Representa las 4 cartas de un jugador.
    Calcula automáticamente sus propiedades (pares, juego) al crearse.
    """
    def __init__(self, cartas: list[Carta]):
        self.cartas = cartas
        self.cartas_ordenadas = sorted(self.cartas, key=lambda c: c.valor_comparacion, reverse=True)
        
        self._valores_comparacion = [c.valor_comparacion for c in self.cartas_ordenadas]
        self._valor_juego_total = sum(c.valor_juego for c in self.cartas)
        self.tiene_juego_bool = self._valor_juego_total >= 31
        
        conteo = Counter(self._valores_comparacion)
        self._pares_info = self._calculate_pares_info(conteo)
        self.tiene_pares_bool = self._pares_info['tipo'] != 'Ninguno'

    def _calculate_pares_info(self, conteo: Counter) -> dict:
        """ Método privado para calcular la información de pares a partir de un conteo. """
        valores_conteo = conteo.values()
        
        if 4 in valores_conteo:
            valor = list(conteo.keys())[0]
            return {'tipo': 'Duples', 'valor_principal': valor, 'valor_secundario': valor}
        if 3 in valores_conteo:
            valor = [v for v, c in conteo.items() if c == 3][0]
            return {'tipo': 'Medias', 'valor_principal': valor, 'valor_secundario': 0}
        if 2 in valores_conteo:
            pares = [v for v, c in conteo.items() if c == 2]
            if len(pares) == 2:
                return {'tipo': 'Duples', 'valor_principal': max(pares), 'valor_secundario': min(pares)}
            else:
                return {'tipo': 'Pares', 'valor_principal': pares[0], 'valor_secundario': 0}
        return {'tipo': 'Ninguno', 'valor_principal': 0, 'valor_secundario': 0}

    def __repr__(self): return f"Mano({[str(c) for c in self.cartas]})"
    def get_valores_comparacion(self) -> list[int]: return self._valores_comparacion
    def tiene_juego(self) -> bool: return self.tiene_juego_bool
    def get_juego_valor(self) -> int: return self._valor_juego_total
    def tiene_pares(self) -> bool: return self.tiene_pares_bool
    def get_pares_info(self) -> dict: return self._pares_info