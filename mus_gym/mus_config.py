# mus_gym/mus_config.py

# --- Constantes de la Baraja ---
PALOS = ['Oros', 'Copas', 'Espadas', 'Bastos']
RANGOS_REALES = ['As', '2', '3', '4', '5', '6', '7', 'Sota', 'Caballo', 'Rey']

# --- Valores para el lance de JUEGO ---
VALORES_JUEGO = {
    'As': 1, '2': 1, '3': 10, '4': 4, '5': 5, '6': 6, '7': 7,
    'Sota': 10, 'Caballo': 10, 'Rey': 10
}

# --- Valores para comparar en GRANDE, CHICA y PARES ---
VALORES_COMPARACION = {
    'As': 1, '2': 1, '3': 10, '4': 4, '5': 5, '6': 6, '7': 7,
    'Sota': 8, 'Caballo': 9, 'Rey': 10
}