"""Enumeraciones del juego."""
from enum import Enum


class MusAction(Enum):
    """Acciones discretas del agente (espacio de acción `Discrete(5)`)."""
    PASO = 0
    ENVIDO = 1      # Apuesta fija de 2 piedras.
    ORDAGO = 2      # Se juega la partida entera en este lance.
    QUIERO = 3
    NO_QUIERO = 4


class MusLance(Enum):
    """Los cuatro lances, en orden de juego."""
    GRANDE = 0
    CHICA = 1
    PARES = 2
    JUEGO = 3


class LanceState(Enum):
    """Estado de la apuesta dentro de un lance."""
    START = 0       # Nadie ha apostado: se puede pasar, envidar u ordagar.
    BET_MADE = 1    # Hay apuesta pendiente de respuesta.
