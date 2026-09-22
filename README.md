# CNN-Para-Reconocimiento-De-Gestos-CoppeliaSim
# Sistema de Control Gestual Robótico en Tiempo Real con CNN

Este repositorio contiene la implementación completa de un sistema de teleoperación gestual para el control de un brazo robótico articulado de 3 Grados de Libertad (GDL) con pinza en el simulador CoppeliaSim, utilizando una Red Neuronal Convolucional (CNN) en PyTorch y una arquitectura de software modular y desacoplada.

---

## 1. Arquitectura del Sistema y Principios de Diseño

El sistema está diseñado bajo el principio fundamental de **separación de responsabilidades**:
1. **Percepción (`preprocess.py`, `model.py`, `cnn_inference.py`):** Desacoplada por completo del simulador. Reconoce de 0 a 4 dedos mostrados frente a una cámara web.
2. **Seguridad Temporal (`command_filter.py`):** Máquina de estados finita (FSM: `IDLE`, `BUSY`, `WAIT_ZERO`) que garantiza que **un cuadro de cámara nunca dispare por sí solo una secuencia de movimientos**.
3. **Actuación (`robot_adapter.py`):** Único módulo que conoce la API remota ZMQ de CoppeliaSim. Traduce los comandos validados a incrementos angulares discretos con límites articulares físicos.


# Sistema de Control Gestual Robótico en Tiempo Real con CNN

Este repositorio contiene la implementación completa de un sistema de teleoperación gestual para el control de un brazo robótico articulado de 3 Grados de Libertad (GDL) con pinza en el simulador CoppeliaSim, utilizando una Red Neuronal Convolucional (CNN) en PyTorch y una arquitectura de software modular y desacoplada.

---

## 1. Arquitectura del Sistema y Principios de Diseño

El sistema está diseñado bajo el principio fundamental de **separación de responsabilidades**:
1. **Percepción (`preprocess.py`, `model.py`, `cnn_inference.py`):** Desacoplada por completo del simulador. Reconoce de 0 a 4 dedos mostrados frente a una cámara web.
2. **Seguridad Temporal (`command_filter.py`):** Máquina de estados finita (FSM: `IDLE`, `BUSY`, `WAIT_ZERO`) que garantiza que **un cuadro de cámara nunca dispare por sí solo una secuencia de movimientos**.
3. **Actuación (`robot_adapter.py`):** Único módulo que conoce la API remota ZMQ de CoppeliaSim. Traduce los comandos validados a incrementos angulares discretos con límites articulares físicos.
[ Cámara Web / ROI ]
│
▼
[ preprocess.py ] ───> Espacio YCrCb + Componentes Conexos + Crop Cuadrado (64x64)
│
▼
[ model.py ] ───> FingerCountCNN (PyTorch - 537k parámetros)
│
▼
[ command_filter.py ] ───> Ventana temporal + Quórum + FSM (IDLE/BUSY/WAIT_ZERO)
│
▼
[ robot_adapter.py ] ───> CoppeliaSim ZMQ API (Control de PArm + PGripStraight) 
---

## 2. Mapeo de Comandos Gestuales

| Clase | Gesto | Acción Robótica | Justificación de Seguridad |
| :---: | :--- | :--- | :--- |
| **0** | Mano cerrada / Sin mano | **PARADA LÓGICA / REPOSO** | Bloquea órdenes activas y limpia el enclavamiento. |
| **1** | 1 Dedo extendido | Mover Articulación 1 (Base) | Giro discreto de $\pm 15^\circ$ con inversión en límites. |
| **2** | 2 Dedos extendidos | Mover Articulación 2 (Hombro) | Movimiento discreto de $\pm 15^\circ$ con límites físicos. |
| **3** | 3 Dedos extendidos | Mover Articulación 3 (Codo) | Flexión discreta de $\pm 15^\circ$ con límites físicos. |
| **4** | 4 Dedos extendidos | Alternar Pinza (Abrir/Cerrar) | Accionamiento por velocidad dinámica con carrera extendida. |

---

## 3. Resumen de Resultados Experimentales

### 3.1. Auditoría del Dataset y Partición Anti-Fuga
- **Imágenes evaluadas:** 15,215 imágenes crudas.
- **Filtrado:** 10,939 imágenes (71.9%) aprobadas por compatibilidad con segmentación de piel; 4,276 descartadas por iluminación deficiente o fondos ruidosos.
- **Partición:** División estricta por sesiones completas e independientes (`data/manifest.csv`) con semilla fija (`SEED=42`):
  - *Train:* 6,793 imágenes (62%)
  - *Val:* 1,443 imágenes (13%)
  - *Test:* 2,703 imágenes (25%) — *Evaluado una sola vez y protegido con `reports/test_done.flag`*.
  - *Comprobación de fuga:* 0 solapamientos de sujetos o sesiones entre subconjuntos.

### 3.2. Estudio Sistemático de Ablación (`reports/ablation.csv`)
Comparación bajo condiciones idénticas de entrenamiento para determinar el mejor modo de preprocesamiento:

| Modo de Entrada | Balanced Accuracy (Val) | F1 Promedio | Conclusión Técnica |
| :--- | :---: | :---: | :--- |
| `mask` (Binaria) | 93.37% | 93.00% | Rápida pero pierde la textura y separación de comisuras. |
| `gray` (Gris) | 94.52% | 94.40% | Conserva textura, pero susceptible a fondos no controlados. |
| **`masked_gray`** | **95.75%** | **95.61%** | **Ganador.** Elimina el fondo ruidoso conservando la textura de los dedos. |

### 3.3. Benchmark de Latencia Real (`latency.py`)
Medición sobre hardware dedicado (**NVIDIA GeForce RTX 4060 Laptop GPU**, 50 warm-up, 500 iteraciones):
- **Preprocesamiento (OpenCV):** Mediana 0.86 ms ($p_{95}$: 1.09 ms).
- **Inferencia CNN (PyTorch / GPU):** Mediana 0.67 ms ($p_{95}$: 1.04 ms).
- **Pipeline Completo por Cuadro:** **1.55 ms** ($p_{95}$: 2.11 ms) $\approx$ **~644 FPS teóricos**.
- **Tamaño del modelo:** **2.10 MB** (537,797 parámetros).

---

## 4. Estructura del Repositorio

```text
├── config.yaml               # Parámetros centralizados (ROI, umbrales, cinemática)
├── preprocess.py             # Pipeline común de visión por computador
├── model.py                  # Definición de la CNN e inspección analítica de parámetros
├── calibrate.py              # Calibrador interactivo de color YCrCb
├── capture.py                # Grabación de imágenes crudas con metadatos
├── filter_external_dataset.py# Auditoría y compatibilidad de dataset externo
├── split_dataset.py          # Partición por sesiones sin data leakage
├── train.py                  # Entrenamiento en GPU con aumentos y Early Stopping
├── evaluate.py               # Evaluación formal (--modo val / --modo test)
├── latency.py                # Medición de tiempos y tamaño del modelo
├── run_ablation.py           # Script de ablación automática
├── command_filter.py         # Filtro temporal y FSM (IDLE / BUSY / WAIT_ZERO)
├── cnn_inference.py          # Clasificador en vivo desacoplado del simulador
├── robot_adapter.py          # Adaptador de CoppeliaSim con API remota ZMQ
├── app.py                    # Aplicación interactiva con GUI y logging
├── record_trials.py          # Grabación de videos de prueba etiquetados
├── run_phase8.py             # Diagnóstico causal y reoptimización offline del filtro
├── run_sequences.py          # Validación autónoma de secuencias de tareas robóticas
├── data/                     # Manifest, logs de captura y datasets crudos
├── models/                   # Pesos guardados (best_model.pth) y metadatos JSON
└── reports/                  # Matrices de confusión, curvas, ablación y reportes
