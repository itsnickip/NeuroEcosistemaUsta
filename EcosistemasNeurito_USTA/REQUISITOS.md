# Verificación de requisitos de la propuesta — EcosistemasNeurito

Integrantes: Michelt Camargo · Nicoll Pérez · Sergio Ávila

La fuente de requisitos es la propuesta académica fechada el 8 de noviembre de 2026. Este mapa indica dónde comprobar cada punto; no sustituye la ejecución de las pruebas ni la lectura del informe. Las rutas `INFORME_TECNICO.*`, `DECISIONES.md` y `results/` corresponden al paquete académico separado `Entregables_EcosistemasNeurito.zip`; `models/` viene incluido en el proyecto y se copia en ese paquete para revisar la evidencia.

| Requisito | Implementación y evidencia |
| --- | --- |
| Entorno 2D, plantas, presas, depredadores, energía, movimiento, captura y fin de episodios | `ecosistemas_neurito/world.py`, `ecosistemas_neurito/viewer.py`, `tests/test_world.py`. |
| Dos tipos de agentes, decisiones efectivamente neuronales | `ecosistemas_neurito/experiments.py::_actions`, `ecosistemas_neurito/viewer.py::advance`; `argmax` de la red produce las acciones. |
| Una red multicapa por especie con al menos dos capas ocultas | `ecosistemas_neurito/network.py` (`RedNeurito`, `relu`, `softmax`); arquitecturas `8→14→16→12→5` (tres ocultas) y `8→16→12→5` (dos ocultas), con 8 entradas distintas por especie en `world.py`. |
| Entrenamiento real y cambio de parámetros | `ecosistemas_neurito/training.py`, `models/entrenamiento.json`, historiales por época, pares inicial/entrenado NPZ, `tests/test_pipeline.py`. |
| Comparación entre dos configuraciones y con referencia sin aprendizaje | `ecosistemas_neurito/experiments.py`, `results/episodios.csv`, `results/resumen.csv`, `INFORME_TECNICO.pdf`. |
| Visualización, estadísticas e inspección de decisiones | `ecosistemas_neurito/viewer.py`; `python -m ecosistemas_neurito view` y clic sobre un agente. |
| Código, dependencias, instalación, semillas y configuración | `README.md`, `pyproject.toml`, `uv.lock`, `results/configuracion.json`, `models/entrenamiento.json`. |
| Tres escenarios y varias semillas por escenario | `world.py::SCENARIOS` y `results/episodios.csv`; ocho semillas emparejadas por caso. |
| Antes/después, dos arquitecturas y referencia | Casos `inicial`, `entrenado`, `arquitectura_b`, `reglas`, `aleatorio` en `experiments.py`; análisis en el informe. |
| Hipótesis, controles, tabla o gráfica, análisis y conclusiones | `INFORME_TECNICO.pdf` secciones 4 a 6; tablas y `results/*.svg`. |
| Informe técnico de máximo diez páginas | `INFORME_TECNICO.pdf` (dos páginas), con Markdown editable y script de reconstrucción. |
| Bitácora de mínimo seis decisiones | `DECISIONES.md` contiene nueve decisiones con alternativas, evidencia y criterio. |
| Declaración y validación del uso de IA generativa | `INFORME_TECNICO.pdf` sección 7 y pruebas automatizadas. |
| Sustentación y demostración en vivo | `SUSTENTACION.md` incluye secuencia, cambios de parámetro y preguntas individuales. |

Antes de la entrega, el equipo debe verificar que el docente pueda acceder al proyecto y al paquete académico separado, según las reglas del curso.
