# Guion de sustentación — EcosistemasNeurito

Integrantes: Michelt Camargo · Nicoll Pérez · Sergio Ávila

Duración sugerida: 8 a 12 minutos. Cada integrante debe ejecutar y explicar una parte; el guion es una ayuda de preparación, no una respuesta para memorizar. El informe, la bitácora y los resultados de la versión analizada están en el paquete académico separado `Entregables_EcosistemasNeurito.zip`.

## Antes de la sesión

1. En la carpeta del proyecto, ejecutar `uv sync` y `uv run python -m unittest discover -s tests -v`.
2. Verificar `models/entrenamiento.json` en el proyecto y `results/episodios.csv` en el paquete académico. Si se modificó el código o la configuración, reconstruir con `train` y `evaluate` y actualizar el paquete antes de citar resultados.
3. Tener abiertos `INFORME_TECNICO.pdf`, `DECISIONES.md` y las dos gráficas SVG de `results/` desde el paquete académico.

## Demostración reproducible

1. Abrir `uv run python -m ecosistemas_neurito view --scenario abundante --seed 101`. Mostrar plantas, ambas especies, energía y capturas. Hacer clic en una presa y en un depredador; alternar las pestañas **Decisión** y **Entradas** para enseñar las cinco probabilidades Softmax, la acción elegida y las 8 entradas de la red.
2. Usar **Reiniciar** para repetir exactamente la semilla y observar la misma trayectoria con las mismas políticas. Cambiar solo una política con los botones del panel y volver a reiniciar; explicar por qué la trayectoria cambia. **Nueva semilla** modifica las posiciones iniciales.
3. Elegir **Escasez** y luego **Presión** en el selector superior. Relacionar visualmente las diferencias con el número de plantas y depredadores. Usar **Pausar** y los controles de velocidad para explicar un momento concreto.
4. Mostrar en `models/entrenamiento.json` el cambio de pesos L2, las métricas de validación y los ejemplos de la acción `quieto` para depredadores; comparar archivos `*_inicial.npz` y `*_entrenado.npz`.
5. Abrir `results/resumen.csv` o las gráficas del paquete académico. Explicar una comparación favorable y una desfavorable: la presa entrenada supera al azar, pero no a las reglas; y en presas la red alternativa (2 ocultas) sobrevive más que la base aunque tenga menor precisión de validación. No afirmar superioridad general con ocho semillas.

## Preguntas individuales probables

| Pregunta | Elemento que se debe poder señalar |
| --- | --- |
| ¿Qué recibe la presa y qué recibe el depredador? | Tuplas `PREY_INPUTS` y `PREDATOR_INPUTS` en `world.py`; ambas tienen 8 valores (capa de entrada de 8 neuronas). |
| ¿Cuántas capas ocultas hay? | `ARCHITECTURES` en `training.py`: base `14-16-12` (tres ocultas, ReLU) y alternativa `16-12` (dos ocultas). Salida Softmax de 5. |
| ¿Qué son época, batch y learning rate aquí? | `EPOCHS = 80` pasadas completas, `BATCH_SIZE = 64` ejemplos por actualización, `LEARNING_RATE = 0.004` en Adam (`training.py`). |
| ¿Cómo se modifican los pesos? | `RedNeurito._backpropagation` calcula los gradientes de la entropía cruzada (usa `relu_derivada`); `RedNeurito.fit` aplica el paso Adam por mini-lote. |
| ¿Por qué esto es aprendizaje supervisado y no refuerzo? | Las etiquetas vienen de `rule_action`; no se optimiza directamente una recompensa del episodio. |
| ¿Puede la interfaz aparentar aprendizaje sin que exista? | Sí, en general. Aquí lo contradicen pesos iniciales/finales diferentes, historiales y `argmax` neuronal conectado al movimiento. |
| ¿Qué es la referencia de comparación? | Reglas y acciones aleatorias; el oponente se mantiene en reglas para aislar la especie evaluada. |
| ¿Qué significa la desviación estándar? | Variación de la métrica entre ocho semillas de simulación, no certeza sobre todos los ecosistemas posibles. |
| ¿Qué resultado contradijo la expectativa? | La presa entrenada no superó la regla que proporcionó etiquetas, y la arquitectura con más precisión (base) no fue la que más sobrevivió; discutir error de imitación y nuevos estados. |
| ¿Qué cambiarían primero? | Recoger estados inducidos por la red de presas, reentrenar y evaluar con otras semillas. |

No cambiar el código durante la sustentación sin registrar qué configuración y semillas corresponden a los resultados mostrados.
