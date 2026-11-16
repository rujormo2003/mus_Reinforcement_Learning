"""Entidades básicas: carta, baraja española de 40 cartas y mano de 4 cartas."""
from __future__ import annotations

import random
from collections import Counter

from .mus_config import PALOS, RANGOS_REALES, VALORES_COMPARACION, VALORES_JUEGO


class Carta:
    """Una carta de la baraja española."""

    def __init__(self, palo: str, rango: str):
        self.palo = palo
        self.rango = rango
        self.valor_juego = VALORES_JUEGO[rango]
        self.valor_comparacion = VALORES_COMPARACION[rango]

    def __repr__(self):
        return f"{self.rango} de {self.palo}"


class Baraja:
    """Baraja de 40 cartas que se puede mezclar y repartir."""

    def __init__(self):
        self.cartas: list[Carta] = []
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
    """Las 4 cartas de un jugador. Precalcula pares y juego al crearse."""

    def __init__(self, cartas: list[Carta]):
        self.cartas = cartas
        self.cartas_ordenadas = sorted(cartas, key=lambda c: c.valor_comparacion, reverse=True)

        self._valores_comparacion = [c.valor_comparacion for c in self.cartas_ordenadas]
        self._valor_juego_total = sum(c.valor_juego for c in cartas)
        self.tiene_juego_bool = self._valor_juego_total >= 31

        self._pares_info = self._calculate_pares_info(Counter(self._valores_comparacion))
        self.tiene_pares_bool = self._pares_info["tipo"] != "Ninguno"

    @staticmethod
    def _calculate_pares_info(conteo: Counter) -> dict:
        """Clasifica la mano en Ninguno / Pares / Medias / Duples."""
        repeticiones = conteo.values()

        if 4 in repeticiones:  # Cuatro iguales cuentan como duples.
            valor = next(iter(conteo))
            return {"tipo": "Duples", "valor_principal": valor, "valor_secundario": valor}
        if 3 in repeticiones:
            valor = next(v for v, c in conteo.items() if c == 3)
            return {"tipo": "Medias", "valor_principal": valor, "valor_secundario": 0}
        pares = [v for v, c in conteo.items() if c == 2]
        if len(pares) == 2:
            return {"tipo": "Duples", "valor_principal": max(pares), "valor_secundario": min(pares)}
        if len(pares) == 1:
            return {"tipo": "Pares", "valor_principal": pares[0], "valor_secundario": 0}
        return {"tipo": "Ninguno", "valor_principal": 0, "valor_secundario": 0}

    def __repr__(self):
        return f"Mano({[str(c) for c in self.cartas]})"

    def get_valores_comparacion(self) -> list[int]:
        return self._valores_comparacion

    def tiene_juego(self) -> bool:
        return self.tiene_juego_bool

    def get_juego_valor(self) -> int:
        return self._valor_juego_total

    def tiene_pares(self) -> bool:
        return self.tiene_pares_bool

    def get_pares_info(self) -> dict:
        return self._pares_info
