"""Constantes de la baraja española (40 cartas, variante de 8 reyes)."""

PALOS = ["Oros", "Copas", "Espadas", "Bastos"]
RANGOS_REALES = ["As", "2", "3", "4", "5", "6", "7", "Sota", "Caballo", "Rey"]

# Valor para sumar el juego (31, 32, ...). Figuras y treses valen 10; doses valen 1.
VALORES_JUEGO = {
    "As": 1, "2": 1, "3": 10, "4": 4, "5": 5, "6": 6, "7": 7,
    "Sota": 10, "Caballo": 10, "Rey": 10,
}

# Valor para comparar cartas en Grande/Chica/Pares.
# Variante de "8 reyes": el 3 cuenta como Rey y el 2 como As.
VALORES_COMPARACION = {
    "As": 1, "2": 1, "3": 10, "4": 4, "5": 5, "6": 6, "7": 7,
    "Sota": 8, "Caballo": 9, "Rey": 10,
}
