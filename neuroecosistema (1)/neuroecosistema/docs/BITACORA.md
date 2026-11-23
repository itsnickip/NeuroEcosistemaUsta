# Bitácora de decisiones técnicas — NeuroEcosistema

Cada decisión indica alternativas, evidencia y criterio. Los números salen de
`resultados/RESULTADOS.md` (versión final, v3) y de `docs/evidencia_v1/` y
`docs/evidencia_v2_ne_pura/` (versiones anteriores, conservadas a propósito).
Semillas de evaluación: 1000–1009, 10 repeticiones por celda.

**Historia del proyecto en una línea por versión**

| Versión | Cambio | Qué pasó | Evidencia |
|---|---|---|---|
| v1 | Neuroevolución pura, costo QUIETO presa 0.0015 | Las redes aprendieron a **no moverse**: en B sobrevivían 0.98 comiendo 1.5 plantas en total. Depredadores estancados en aptitud 0.299 (quietos) | `docs/evidencia_v1/` |
| v2 | Costos corregidos + término de acercamiento | Presas aprenden (A: 0.28 → 0.41); depredadores **0 capturas** frente a presas por reglas | `docs/evidencia_v2_ne_pura/` |
| v3 | Imitación supervisada + neuroevolución | Depredadores 0 → 4.0 capturas (solo imitación). La evolución posterior **empeoró** la red imitada | `resultados/` |

---

## D1. Representación del entorno: qué percibe cada agente

**Decisión.** Entradas relativas y normalizadas al objetivo visible más cercano
(dx, dy, distancia ÷ radio de visión), energía propia, posición en [-1, 1] y un
vector hacia el obstáculo más cercano que crece al acercarse. Presa: 11 entradas
(planta + depredador). Depredador: 9 (presa + conteo de presas visibles). Si no
hay nada a la vista: dx = dy = 0 y distancia = 1.

**Alternativas.** (a) Coordenadas absolutas de todos los agentes: tamaño
variable y la red tendría que aprender la resta. (b) Rejilla/“retina” de
sensores por sectores (p. ej. 8 rayos): más rica pero 3–5× más entradas y
parámetros. (c) Solo el objetivo más cercano (elegida).

**Excluidas y por qué.** Velocidad/dirección de los demás (la presa no puede
anticipar la persecución → limitación conocida); el 2.º depredador más cercano
(en C con 5 depredadores la presa puede huir de uno hacia otro); otros
depredadores para el depredador (no hay coordinación); el paso de tiempo (no
debe depender del reloj del episodio).

**Evidencia.** Con estas entradas la política por reglas —que usa exactamente
el mismo vector— logra 0.75 de supervivencia en A y 4.4 capturas por
depredador, así que la información alcanza para un comportamiento competente.
La imitación del depredador llega a 79 % de exactitud: la regla es casi una
función de (dx, dy). La de la presa solo 56 %, porque cuando no ve nada la regla
deambula al azar (no es predecible desde la entrada).

**Criterio.** Entrada fija, pequeña y comparable entre especies; misma
información para red y referencia (comparación justa).

## D2. Diseño de las redes

**Decisión.** MLP propio en NumPy, dos capas ocultas tanh, salida softmax de 9
acciones (QUIETO + 8 direcciones), acción = argmax.
- `base`: presa 11→16→12→9 (513 parámetros); depredador 9→16→8→9 (377).
- `pequena`: 11→6→4→9 y 9→6→4→9.

**Alternativas.** Salida continua (ángulo + velocidad) con tanh: más natural
pero complica la imitación (regresión de ángulos con discontinuidad en ±π).
ReLU: en neuroevolución las neuronas muertas no se recuperan por gradiente;
tanh además acota activaciones (útil para el inspector visual).

**Evidencia (E1/E2, arquitectura base vs pequeña).** Presas: sin diferencia
significativa en ningún escenario (A +0.004, p = 0.87; B −0.026, p = 0.18;
C −0.041, p = 0.25). Depredadores: la pequeña capturó **más** (A 4.35 vs 1.85,
p ≈ 3e-6; B 3.00 vs 2.15, p ≈ 0.001; C sin diferencia). Contradice la
expectativa de “más neuronas = mejor”.

**Interpretación.** El problema del depredador es casi lineal (ir hacia dx, dy);
con menos parámetros la mutación de σ = 0.10 perturba menos la conducta
heredada de la imitación (ver D5). `historial.csv` lo respalda: al final de la
ronda 0 el mejor depredador base se alejó 8.4 (norma L2) de su punto de
partida; el pequeño solo 4.4. La capacidad no es el cuello de botella.

**Criterio.** Se mantiene `base` como modelo principal por ser el que se
planeó antes de ver datos (evitar elegir a posteriori), y se reporta la
pequeña como comparación de arquitecturas.

## D3. Mecanismo de aprendizaje: imitación + neuroevolución

**Decisión.** Dos etapas: (1) aprendizaje supervisado imitando la política por
reglas (40 000 / 31 000 ejemplos, 80 épocas, entropía cruzada, retropropagación,
mini-lotes de 256, tasa 0.15); (2) neuroevolución (población 24, 35
generaciones × 2 rondas, 3 élites, torneo 3, cruce uniforme 0.5, mutación
gaussiana σ = 0.10) partiendo de copias mutadas de la red imitada.

**Alternativas.** Neuroevolución pura (v1, v2); solo imitación; aprendizaje
por refuerzo con gradiente de política (requiere varianza muy alta con episodios
de 400 pasos y recompensas escasas).

**Evidencia.**
- v2 (neuroevolución pura): depredadores con 0.0 capturas en A, B y C frente a
  presas por reglas. En la ronda 1 sí capturaban, pero solo contra presas
  neuronales torpes → sobreajuste al oponente.
- v3 solo imitación: 4.00 / 3.90 / 2.92 capturas por depredador (A/B/C),
  estadísticamente igual a las reglas (4.40 / 4.10 / 2.62).
- v3 imitación + evolución (“red base”): 1.85 / 2.15 / 1.44, **peor** que solo
  imitación en los tres escenarios (p < 1e-5). Lo mismo en presas en A
  (0.613 vs 0.688, p ≈ 0.01) y B (0.418 vs 0.453, p ≈ 0.04).

**Cómo cambian los pesos (evidencia de entrenamiento real).**
`modelos/<arq>/historial.csv` registra por generación la norma de los pesos del
mejor, su distancia a la generación 0 y el cambio respecto a la anterior;
`entrenamiento.png` lo grafica (la distancia del mejor depredador base a la
generación 0 llega a 8.4 en la ronda 0 y a 13.4 en la ronda 1).
`imitacion.png` muestra la pérdida bajando de 2.1 a 0.50 (depredador).

**Criterio.** La imitación resolvió el arranque en frío (el problema
principal de v2). La etapa evolutiva se conserva porque es la que permitiría
superar a la regla, pero los datos dicen que, con esta configuración, no lo
logró (ver D5 y D9).

## D4. Función de aptitud (criterio optimizado) y comportamientos indebidos

**Decisión.**
- Presa: `supervivencia + 0.05 · plantas_por_presa`.
- Depredador: `capturas_por_depredador + 0.5 · acercamiento + 0.2 · supervivencia`.

**Comportamientos que favorece indebidamente (observados, no hipotéticos).**
1. *Inmovilidad (v1).* Con costo de quedarse quieto 0.0015, una presa quieta
   vivía 0.70 / 0.0015 = 466 pasos > 400. Las redes lo explotaron (B:
   supervivencia 0.98, 1.5 plantas comidas en total). Corrección: 0.0025 presa
   (280 pasos) y 0.003 depredador (266 pasos): ahora sin comer se muere.
2. *Ahorro sin cazar (v1).* La aptitud del depredador quedó plana en 0.299 =
   lo que da sobrevivir quieto. Corrección: término de acercamiento (shaping).
3. *Escoltar sin capturar (v3).* El shaping premia estar cerca. La red base
   evolucionada mantiene el mismo acercamiento que la imitada (A 0.62 vs 0.62)
   pero captura la mitad (1.85 vs 4.00) y muere más (vivos al final 0.30 vs
   0.60). Es la firma del riesgo anticipado: acercarse da puntos sin cerrar.
4. *Atascarse en la esquina (v3, observado en `simular.py`, escenario C,
   semilla 1000, paso 150).* Sin presas a la vista, el depredador base con
   pos_x = pos_y = +1 elige SE (58.6 %): empuja contra la esquina y gasta
   energía sin explorar. La regla de deambular empuja al centro; la red no
   heredó bien ese caso porque, en la imitación, “deambular” es azaroso y no
   predecible desde la entrada (exactitud de imitación de la presa: 56 %).
5. *Huir sin comer (presa).* El peso 0.05 de la comida es bajo: la red base
   come 64.8 plantas en A frente a 159.1 de la imitada y 211 de las reglas.

**Criterio.** La supervivencia es el objetivo declarado de la presa; la captura
el del depredador. Los términos secundarios se mantienen pequeños, y los
riesgos se miden explícitamente con métricas separadas (comida, acercamiento,
capturas) en lugar de confiar en la aptitud agregada.

## D5. Ruido de evaluación en la neuroevolución

**Decisión.** 3 episodios por genoma (uno por escenario), mismas semillas para
todos los genomas de una generación (comparación justa), semillas nuevas en
cada generación (evita sobreajustar un mapa), selección final entre élites con
6 episodios de validación (semillas ≥ 90 000).

**Alternativas.** Más episodios por genoma (menos ruido, más costo lineal);
semillas fijas para toda la evolución (menos ruido, sobreajuste); re-evaluar
élites cada generación.

**Evidencia.** Las tres élites finales del depredador base en la ronda 1
obtuvieron 3.60 / 2.97 / 4.62 en validación: diferencias del mismo tamaño que
la mejora buscada. En la ronda 0 la aptitud del “mejor” del depredador base
osciló entre 3.55 y 6.14 de una generación a otra, y la **aptitud media de la
población bajó** de 4.21 a 2.58 (presa base: 0.73 → 0.54). Es decir, la
población empezó concentrada en el óptimo de imitación, la mutación la
dispersó a regiones peores y la selección, con este nivel de ruido, no logró
retenerla (“maldición del ganador”: se elige en buena parte por suerte del
episodio). Mientras tanto, la distancia de los pesos a la generación 0 crece
de forma sostenida aunque la aptitud no mejore.

**Criterio.** Presupuesto de cómputo: ~12 min por arquitectura en 2 núcleos.
Es la decisión que se modificaría primero (D9).

## D6. Diseño experimental

**Decisión.** Tres experimentos, cada uno con una única variable independiente:
- **E1** varía la política de las presas; depredadores fijos = reglas.
- **E2** varía la política de los depredadores; presas fijas = reglas.
- **E3** ecosistema completo: redes sin entrenar vs redes entrenadas vs reglas.

Controladas: parámetros del mundo, escenario, semillas 1000–1009 (iguales para
todas las políticas: mismo mapa, mismas posiciones iniciales), oponente.
Las semillas de evaluación nunca se usan al entrenar (≥ 10 000, ≥ 20 000 para
imitación, ≥ 90 000 validación). Se reporta media ± desviación estándar y la t
de Welch (orientativa, n = 10).

**Comparaciones obligatorias cubiertas.** Antes/después (sin entrenar →
imitación → imitación + evolución); dos arquitecturas (base vs pequeña); dos
referencias (aleatoria y reglas).

**Alternativa descartada.** Medir siempre en E3 (red vs red): las dos especies
cambian a la vez y no se puede atribuir el efecto a ninguna (en E3 B la
supervivencia de presas *baja* tras entrenar, 0.551 → 0.454, porque mejoraron
los depredadores, no porque empeoraran las presas).

## D7. Reglas del mundo: sin reproducción, depredador 10 % más rápido

**Decisión.** Poblaciones fijas (sin nacimientos), episodios de 400 pasos,
velocidad presa 1.0 y depredador 1.1, plantas que reaparecen en otro lugar.

**Alternativas.** Reproducción por energía (dinámica Lotka–Volterra más
realista); misma velocidad para ambos.

**Evidencia.** Con reproducción, la métrica “supervivencia” deja de comparar lo
mismo entre políticas (más nacimientos = más agentes expuestos). Con velocidad
igual, una presa que huye en línea recta nunca es alcanzada y las capturas
dependerían solo de acorralar en bordes. Con 1.1, las reglas capturan 4.4 por
depredador en A y la referencia aleatoria 0: hay rango medible para aprender.

**Criterio.** Métricas comparables por episodio por encima de realismo
ecológico. Limitación: no se estudia el equilibrio poblacional a largo plazo.

## D8. Referencias: reglas con las mismas entradas que la red

**Decisión.** La política por reglas lee el mismo vector de observación (no
el estado interno del mundo): presa = huir si el depredador está a < 0.65 del
radio de visión, si no ir a la planta, si no deambular; depredador = ir hacia
la presa visible, si no deambular. Ambas con repulsión de obstáculos.

**Evidencia.** En la primera prueba la regla se atascaba tras obstáculos en C
(comía 19.8 plantas vs 242 en A). Se normalizaron los vectores (la atracción
se volvía minúscula al acercarse) y se añadió deslizamiento contra obstáculos
en el mundo; después la presa por reglas en C come 61.5 plantas y los
depredadores por reglas en C pasaron de 1.2 a 13.1 capturas totales.

**Criterio.** Una referencia fuerte y justa: si la red la superara, sería por
cómo usa la información, no por tener más.

## D9. Mejora propuesta (qué cambiaríamos primero)

**Decisión a cambiar: la etapa evolutiva (D5), no la arquitectura.**

**Evidencia que la sustenta.** (1) La imitación sola iguala a las reglas en
depredadores y la evolución la degrada en 3/3 escenarios con p < 1e-5. (2) La
aptitud media de la población cae durante la evolución (depredador base
4.21 → 2.58), lo que señala al operador de mutación/selección, no a la red.
(3) La red pequeña, que se desplazó la mitad en el espacio de pesos (4.4 vs
8.4), terminó mejor (4.35 vs 1.85 capturas en A). (4) La varianza de la aptitud entre élites (2.97 –
4.62) es del tamaño de las diferencias que la selección intenta detectar.
(5) El patrón “mismo acercamiento, menos capturas” apunta al shaping (D4).

**Cambios concretos, en orden, cada uno como experimento controlado:**
1. `--sigma-mutacion 0.03` (perturbar menos la solución imitada).
2. `--episodios 6` (dos por escenario: menos ruido en la selección).
3. Incluir la red imitada como candidata en la validación final, para que la
   evolución no pueda entregar algo peor que su punto de partida.
4. Bajar el peso del acercamiento (0.5 → 0.2) o anularlo tras la imitación,
   ya que la imitación resuelve el arranque en frío que motivó el shaping.

Ejemplo: `uv run entrenar.py --sigma-mutacion 0.03 --episodios 6 --salida modelos_exp`
y `uv run evaluar.py --modelos modelos_exp --salida resultados_exp`.

---

## Resumen para la pregunta orientadora

> ¿En qué condiciones los agentes neuronales desarrollan búsqueda,
> persecución y evasión que mejoran su desempeño frente a estrategias sin
> aprendizaje?

- **Frente a la referencia aleatoria y frente a la red sin entrenar**: siempre,
  en los tres escenarios y para ambas especies (p. ej. presas A 0.613 vs 0.261;
  depredadores B 2.15 vs 0.00 capturas).
- **Frente a la referencia por reglas**: ninguna red la supera de forma
  significativa. Las redes imitadas la igualan (depredadores A–C, presas C).
- **Condición clave observada**: el aprendizaje solo aparece cuando la señal de
  entrenamiento es densa (imitación) o cuando la aptitud no premia atajos
  (v1). Con recompensas escasas y pocas evaluaciones, la neuroevolución pura no
  descubre la persecución (v2) y degrada una solución buena (v3).
