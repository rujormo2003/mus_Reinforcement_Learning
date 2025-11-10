# mus_gym/mus_enums.py
from enum import Enum

class MusAction(Enum):
    """ Acciones discretas que el agente puede tomar. """
    PASO = 0
    ENVIDO = 1
    ORDAGO = 2
    QUIERO = 3
    NO_QUIERO = 4

class MusLance(Enum):
    """ Los cuatro lances del juego. """
    GRANDE = 0
    CHICA = 1
    PARES = 2
    JUEGO = 3

class LanceState(Enum):
    """ El estado de la apuesta dentro de un lance. """
    START = 0
    BET_MADE = 1