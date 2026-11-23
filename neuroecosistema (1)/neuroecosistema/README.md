# NeuroEcosistema — Depredadores y presas

Proyecto final integrador · Electiva I de Profundización (Inteligencia Artificial) · USTA Tunja.

Simulación 2D en la que **presas** y **depredadores** deciden cómo moverse con
**redes neuronales multicapa** escritas en NumPy (sin frameworks de IA),
pre-entrenadas por **imitación supervisada** (retropropagación) y ajustadas por
**neuroevolución** (algoritmo genético sobre los pesos). Se
comparan contra referencias **aleatorias** y **por reglas** en tres escenarios.

## Instalación

Con [uv](https://docs.astral.sh/uv/) (recomendado, igual que en los talleres):

```bash
uv sync
```

Con pip:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python ≥ 3.10. Dependencias: `numpy`, `matplotlib`, `pygame`.

## Ejecución

| Paso | Comando | Qué produce |
|---|---|---|
| 1. Entrenar arquitectura base | `uv run entrenar.py` | `modelos/base/` (sin entrenar, solo imitación, final) |
| 2. Entrenar arquitectura pequeña | `uv run entrenar.py --arquitectura pequena` | `modelos/pequena/` |
| 3. Experimentos | `uv run evaluar.py` | `resultados/RESULTADOS.md`, CSV y PNG |
| 4. Ver la simulación | `uv run simular.py` | ventana pygame |
| Pruebas | `uv run python -m unittest discover -s tests -v` | 15 pruebas |

El repositorio ya incluye los modelos entrenados y los resultados, así que
`simular.py` y `evaluar.py` funcionan sin volver a entrenar. Entrenar cada
arquitectura toma ~10–15 min en un portátil de 2 núcleos (`--workers N` usa más
procesos; `--rapido` hace una prueba de 1 minuto).

### Simulador

```bash
uv run simular.py --escenario C --presa "red base" --depredador reglas
```

- **Clic** sobre un agente: muestra sus 11 (presa) o 9 (depredador) entradas,
  las activaciones de las dos capas ocultas y las probabilidades softmax de las
  9 acciones. La acción ejecutada es el argmax de esa salida.
- `ESPACIO` pausa · `→` un paso · `+/-` velocidad · `1/2/3` escenario
- `P` / `D` cambian la política de presas / depredadores (red base, solo
  imitación, sin entrenar, red pequeña, reglas, aleatoria) · `R` reinicia · `G` otra semilla ·
  `V` radios de visión

## Estructura

```
neuroeco/
  config.py        parámetros del mundo, escenarios A/B/C, arquitecturas, config. de entrenamiento
  mundo.py         simulación 2D vectorizada: movimiento, energía, comer, captura, obstáculos, sensores
  red.py           MLP NumPy (≥2 capas ocultas, tanh, softmax), genoma plano, guardar/cargar .npz
  politicas.py     PoliticaRed, PoliticaReglasPresa/Depredador, PoliticaAleatoria
  simulacion.py    episodio completo + funciones de aptitud
  imitacion.py     pre-entrenamiento supervisado: datos (observación, acción de la regla) + backprop
  evolucion.py     algoritmo genético (élites, torneo, cruce uniforme, mutación gaussiana)
  experimentos.py  experimentos E1/E2/E3, resumen, t de Welch, reporte Markdown
  graficas.py      gráficas de entrenamiento y de resultados
entrenar.py        CLI de entrenamiento (co-evolución por rondas)
evaluar.py         CLI de experimentos
simular.py         visualización interactiva
tests/             pruebas automáticas
modelos/           redes entrenadas, pesos iniciales, historial y config por arquitectura
resultados/        evidencia experimental (episodios.csv, resumen.csv, RESULTADOS.md, PNG)
docs/BITACORA.md   bitácora de decisiones técnicas
docs/evidencia_v1/ versión 1: costo de quedarse quieto mal calibrado (explotado por las redes)
docs/evidencia_v2_ne_pura/ versión 2: neuroevolución pura (los depredadores no aprenden a cazar)
```

## Diseño en una página

**Mundo.** 100×100 unidades, con bordes. Plantas que reaparecen en otro lugar
tras `regeneracion_plantas` pasos. Cada paso: moverse (o quedarse quieto) cuesta
energía; comer planta +0.30; capturar presa +0.45 al depredador; energía 0 =
muerte. Episodio: 400 pasos o hasta que no queden presas. Sin reproducción
(ver bitácora D7).

**Acciones (ambas especies).** 9 salidas: `QUIETO` + 8 direcciones (cada 45°).
Velocidad presa 1.0, depredador 1.1.

**Entradas.** Vectores relativos al objetivo visible más cercano, normalizados
por el radio de visión (0 si no se ve nada; distancia = 1).

| Presa (11) | Depredador (9) |
|---|---|
| planta_dx, planta_dy, planta_dist | presa_dx, presa_dy, presa_dist |
| depred_dx, depred_dy, depred_dist | presas_visibles (conteo/5) |
| energia | energia |
| pos_x, pos_y (en [-1, 1], para los bordes) | pos_x, pos_y |
| obst_dx, obst_dy (crece al acercarse) | obst_dx, obst_dy |

**Redes.** `base`: presa 11→16→12→9, depredador 9→16→8→9. `pequena`:
11→6→4→9 y 9→6→4→9. Capas ocultas tanh, salida softmax, acción = argmax.
Todas las presas de un episodio comparten la misma red (política de especie).

**Entrenamiento (dos etapas).**

1. *Imitación supervisada*: se juegan 10 episodios por escenario con las
   políticas por reglas, se guardan pares (observación, acción de la regla) y la
   red se entrena 80 épocas con entropía cruzada y retropropagación (mini-lotes
   de 256, tasa 0.15). Se guarda como `<especie>_imitacion.npz`.
2. *Neuroevolución*: la población inicial son copias mutadas de la red
   imitada (el individuo 0 sin mutar). Población de 24 genomas (vectores de pesos),
35 generaciones por ronda, 3 élites, torneo de 3, cruce uniforme (p = 0.5),
mutación gaussiana σ = 0.10. Cada genoma se evalúa en 3 episodios (uno por
escenario) con las **mismas semillas para toda la generación**.
   `--inicializacion aleatoria` reproduce la neuroevolución pura (versión 2). Ronda 0: cada
especie contra la referencia por reglas; ronda 1: contra una mezcla de reglas y
la mejor red de la otra especie (co-evolución). Al final se elige entre las
élites con semillas de validación (≥ 90 000) no usadas al entrenar.

**Aptitud.**
- Presa: `supervivencia + 0.05 · plantas_por_presa`
- Depredador: `capturas_por_depredador + 0.5 · acercamiento + 0.2 · supervivencia`

**Escenarios.**

| | Presas | Depredadores | Plantas | Regeneración | Obstáculos |
|---|---|---|---|---|---|
| A — recursos suficientes | 20 | 2 | 45 | 50 pasos | 0 |
| B — escasez de alimento | 20 | 2 | 12 | 140 pasos | 0 |
| C — presión de depredación | 20 | 5 | 35 | 60 pasos | 6 |

## Reproducibilidad

- Toda aleatoriedad pasa por `numpy.random.default_rng(semilla)`; la misma
  semilla genera el mismo mapa y episodio (`tests/` lo verifica).
- Semillas: red inicial `semilla·1000` (+500 depredador); datos de imitación `20 000–20 009`; episodios de entrenamiento
  `≥ 10 000` (registradas por generación en `historial.csv`); validación
  `≥ 90 000`; evaluación `1000–1009`.
- `modelos/<arq>/config.json` y `resultados/config.json` guardan la
  configuración completa usada. Los `.npz` incluyen arquitectura y metadatos.
- Para repetir todo: `uv run entrenar.py && uv run entrenar.py --arquitectura pequena && uv run evaluar.py`.

## Uso de herramientas de IA generativa

> **Completar por el equipo** (requisito 8 de la propuesta): qué herramientas
> se usaron, para qué, qué recomendaciones se aceptaron o rechazaron y qué
> validaciones hicieron (p. ej. las pruebas de `tests/`, la revisión de las
> tablas de `resultados/` y la reproducción de un entrenamiento).
