# Entorno de Mus para Gymnasium y sus experimentos

`mus-gym` es una implementación del juego de cartas español **Mus**, diseñada como un entorno compatible con **Gymnasium** (anteriormente OpenAI Gym). Este proyecto permite el entrenamiento de agentes de Aprendizaje por Refuerzo (RL) utilizando algoritmos estándar (como PPO, DQN, etc.) para aprender a jugar al Mus.

El entorno soporta desde partidas contra bots heurísticos hasta entrenamiento avanzado mediante *Self-Play* y *Curriculum Learning*.

## 📋 Características

* **Reglas Completas:** Implementación de los 4 lances: Grande, Chica, Pares y Juego (incluyendo la lógica de "Punto" dentro de la fase de Juego).
* **Espacio de Acción Discreto:** 5 acciones simplificadas (Paso, Envido, Órdago, Quiero, No Quiero).
* **Observación Vectorial:** El agente recibe un vector de características (tamaño 8) que resume su mano y el estado de la mesa.
* **Entornos Variados:** Desde juego básico hasta entrenamiento competitivo contra modelos congelados.
* **Bot Heurístico:** Incluye un `HeuristicBot` ajustable para evaluar el rendimiento base.

## 📦 Dependencias

El proyecto requiere las siguientes librerías de Python:

* `gymnasium`
* `numpy`
* `sb3_contrib` (para el uso de RecurrentPPO en entornos de self-play)

## 🚀 Instalación y Uso

Asegúrate de tener la carpeta `mus_gym` en tu directorio de trabajo.

### Ejemplo Básico (Jugar contra el Bot Heurístico)

El entorno `Mus-v1` está diseñado para entrenar a un agente (Jugador 0) contra el bot programado (Jugador 1).

```python
import gymnasium as gym
import mus_gym  # Importante para registrar los entornos

# Crear el entorno donde el oponente es el Bot Heurístico
env = gym.make("Mus-v1", target_score=30)

observation, info = env.reset()
terminated = False

while not terminated:
    # Aquí iría tu agente. Por ahora, tomamos una acción aleatoria.
    action = env.action_space.sample() 
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        print("Juego terminado.")
        print("Puntuaciones:", info.get('scores'))

env.close()

```

## Environments (Entornos Disponibles)

El paquete registra los siguientes IDs en Gymnasium:

| ID | Clase | Descripción |
| --- | --- | --- |
| **`Mus-v0`** | `MusEnvironment` | Entorno base PvP. Requiere gestión manual de ambos jugadores. Útil para evaluación o debug. |
| **`Mus-v1`** | `MusEnvConBots` | **Recomendado para empezar.** El agente (J0) juega contra un `HeuristicBot` (J1). |
| **`Mus-v2`** | `MusEnvCurriculum` | El agente juega contra una política (modelo) de RL congelada. Requiere pasar el argumento `opponent_policy`. |
| **`Mus-v3`** | `MusEnvSelfPlay` | Entorno para *Self-Play*. Permite cargar y actualizar dinámicamente el modelo del oponente durante el entrenamiento. |

## 🧠 Espacios de Observación y Acción

### Espacio de Acción (Discrete)

El agente puede tomar una de las siguientes 5 acciones (`mus_enums.MusAction`):

* `0`: **PASO**
* `1`: **ENVIDO** (Apuesta base de 2 piedras)
* `2`: **ÓRDAGO** (Apuesta el juego completo)
* `3`: **QUIERO** (Aceptar apuesta)
* `4`: **NO_QUIERO** (Rechazar apuesta)

### Espacio de Observación (Box)

Un vector de **8 valores flotantes** (`mus_observation.py`):

1. **Reyes:** Proporción de reyes en mano (0.0 a 1.0).
2. **Ases:** Proporción de ases en mano.
3. **Pares:** Tipo de pares (0=Nada, 0.33=Pares, 0.66=Medias, 1.0=Duples).
4. **Juego:** Fuerza del juego (calculada según jerarquía 31, 32, 40...).
5. **Lance Actual:** (Grande, Chica, Pares o Juego).
6. **Estado Apuesta:** (Inicio o Apuesta ya realizada).
7. **Soy Mano:** (1.0 si es mano, 0.0 si es postre).
8. **Diferencia de Puntuación:** (Score Propio - Score Rival) normalizado.

## 📂 Estructura del Código

* **`mus_env.py`**: Lógica central del juego, gestión de turnos y puntuaciones.
* **`mus_env_v2.py`**: Wrapper para entrenamiento contra bots (Single Player perspective).
* **`mus_bot.py`**: Lógica del `HeuristicBot` (reglas fijas y probabilidades).
* **`mus_entities.py`**: Clases `Carta`, `Baraja` y `Mano`.
* **`mus_rules.py`**: Lógica para determinar ganadores de lances (comparación de cartas).
* **`mus_observation.py`**: Construcción del vector de observación para la IA.
* **`mus_selfplay_env.py`**: Lógica avanzada para cargar modelos `.zip` de Stable Baselines como oponentes.

## 🤝 Contribuciones

Si deseas mejorar la lógica del bot, refinar las observaciones o añadir soporte para 4 jugadores (2vs2), ¡siéntete libre de editar `mus_env.py`!

---