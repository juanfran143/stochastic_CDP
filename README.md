# Stochastic Capacitated Dispersion Problem (S-CDP)

Este repositorio implementa una **simheurística forward–backward** para el Stochastic Capacitated Dispersion Problem (S-CDP). El objetivo principal es seleccionar un subconjunto de vértices que maximice la **distancia mínima** entre cualquier par elegido mientras se respetan **capacidades** y, bajo el enfoque epsilon-constraint, se controla el número de colores (tipos) permitidos en la solución.

## Objetivo del proyecto

- **Qué resolvemos:** Dado un grafo completamente ponderado, buscamos el subconjunto de nodos más disperso (maximizar la distancia mínima) sujeto a una capacidad mínima agregada y, opcionalmente, a un límite en la cantidad de tipos de nodo (colores) activos.
- **Por qué es estocástico:** Las capacidades pueden variar; la meta es obtener soluciones robustas que mantengan buena dispersión aun cuando los parámetros cambian.
- **Cómo lo abordamos:** Integrando construcción sesgada, búsqueda tabú y múltiples reinicios, más un barrido con restricciones epsilon para generar un frente de Pareto de (dispersión, diversidad de colores).

## Componentes clave

- `CDP/Main.py`: orquesta la carga de instancias, ejecuta cada caso de prueba con distintas restricciones epsilon, dispara la construcción y mejora de soluciones, y genera resúmenes y gráficas del frente epsilon.【F:CDP/Main.py†L87-L132】【F:CDP/Main.py†L228-L307】
- `CDP/ConstructiveHeuristic.py`: genera soluciones iniciales, gestiona el límite de colores (`epsilon`) y calcula candidatos para la búsqueda local.【F:CDP/ConstructiveHeuristic.py†L1-L117】
- `CDP/LocalSearches.py`: aplica una búsqueda tabú para intensificar alrededor de cada construcción (no mostrado aquí, pero llamado desde `Main.py`).
- `CDP/Instance.py`: define el formato de entrada y prepara distancias, capacidades y paletas de color para cada instancia.【F:CDP/Instance.py†L1-L66】
- `CDP/objects.py`: contiene las estructuras de datos ligeras (`Candidate`, `Edge`, `TestCase`) usadas en toda la tubería.【F:CDP/objects.py†L8-L46】

## Formato de instancias

Cada archivo en `Instances/` define un grafo y sus capacidades:
1. Primera línea: número de nodos.
2. Segunda línea: capacidad mínima requerida.
3. Tercera línea: lista tabulada (`\t`) con la capacidad de cada nodo.
4. Resto de líneas: matriz de distancias (tabulada) entre nodos; se almacenan las aristas con distancia positiva y se ordenan para seleccionar la arista inicial más larga.【F:CDP/Instance.py†L37-L53】

## Flujo de ejecución

1. **Preparar entorno**
   ```bash
   python -m pip install -r CDP/requirements.txt
   ```

2. **Configurar experimentos**
   - Edita `test/run.json` para listar los casos a ejecutar (instancia, semilla, tiempo máximo, pesos beta, iteraciones y `max_epsilon`).【F:test/run.json†L1-L13】
   - Las instancias se resuelven en orden, variando `epsilon` desde 1 hasta `max_epsilon` (o el número de colores disponibles).

3. **Lanzar la simulación**
   ```bash
   python CDP/Main.py
   ```
   - Se cargan las instancias desde `Instances/` (o rutas absolutas), se asigna una paleta de colores determinística y se evalúa cada valor de `epsilon`.【F:CDP/Main.py†L87-L122】【F:CDP/Main.py†L234-L265】

4. **Resultados**
   - `output/deterministic_summary.txt`: resumen de la mejor solución determinista (dispersión máxima alcanzada, tiempo, capacidad).【F:CDP/Main.py†L272-L285】
   - `output/epsilon_summary.txt`: historial completo de dispersiones logradas para cada `epsilon`.【F:CDP/Main.py†L286-L300】
   - Si `plot` es `true` en el caso de prueba, se genera `output/<instancia>_epsilon_frontier.png` con el frente de Pareto comprimido y monotónico.【F:CDP/Main.py†L301-L307】

## Estructura del código

- `CDP/`: núcleo de la simheurística y de las rutinas de entrada/salida.
- `Instances/`: benchmarks base (GKD-b...).
- `test/`: configuración de ejecuciones reproducibles.
- `output/`: se llena automáticamente con los resúmenes y gráficos generados.

## ¿Cómo extenderlo?

- Ajusta las heurísticas de construcción y las reglas tabú en `ConstructiveHeuristic.py` y `LocalSearches.py` para explorar nuevos criterios.
- Para nuevos datasets, coloca el archivo en `Instances/` y agrégalo en `test/run.json` con sus parámetros.
- El límite de colores (`epsilon`) puede usarse para estudiar el intercambio entre dispersión y diversidad: el frente resultante sirve para seleccionar la solución más conveniente según la aplicación.
