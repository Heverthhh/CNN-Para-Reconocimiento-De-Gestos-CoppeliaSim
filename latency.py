"""latency.py - Medición rigurosa de latencia (preprocess, CNN y total) y peso del modelo."""

import os
from pathlib import Path
import platform
import time
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import yaml

from model import FingerCountCNN
from preprocess import preprocess

CONFIG_PATH = "config.yaml"
MODEL_WEIGHTS_PATH = Path("models/best_model.pth")
WARMUP_RUNS = 50
TEST_RUNS = 500


def get_processor_name() -> str:
    """Detecta el nombre formal de la CPU."""
    if platform.system() == "Windows":
        return platform.processor()
    return "CPU Desconocida"


def main() -> None:
    if not MODEL_WEIGHTS_PATH.exists():
        print(f"Error: Modelo no encontrado en {MODEL_WEIGHTS_PATH}.")
        return

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw_name = torch.cuda.get_device_name(0) if device.type == "cuda" else get_processor_name()

    model = FingerCountCNN(num_classes=5).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.eval()

    # Frame sintético típico de webcam (640x480x3)
    dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    print("\n" + "=" * 70)
    print(" BENCHMARK DE LATENCIA Y EFICIENCIA TEMPORAL")
    print(f" Dispositivo: {device.type.upper()} ({hw_name})")
    print(f" Warm-up: {WARMUP_RUNS} pasadas | Muestras de prueba: {TEST_RUNS} ejecuciones")
    print("=" * 70)

    # 1. Calentamiento (Warm-up)
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            x_np, _, _ = preprocess(dummy_frame, cfg)
            x_t = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
            _ = model(x_t)
            if device.type == "cuda":
                torch.cuda.synchronize()

    times_prep = []
    times_cnn = []
    times_total = []

    # 2. Medición desacoplada
    with torch.no_grad():
        for _ in range(TEST_RUNS):
            t0 = time.perf_counter()

            # Fase Preprocesamiento (OpenCV / CPU)
            x_np, _, _ = preprocess(dummy_frame, cfg)
            t1 = time.perf_counter()

            # Fase Inferencia CNN (PyTorch)
            x_t = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
            _ = model(x_t)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            times_prep.append((t1 - t0) * 1000.0)
            times_cnn.append((t2 - t1) * 1000.0)
            times_total.append((t2 - t0) * 1000.0)

    # Métricas estadísticas
    prep_med, prep_p95 = float(np.median(times_prep)), float(np.percentile(times_prep, 95))
    cnn_med, cnn_p95 = float(np.median(times_cnn)), float(np.percentile(times_cnn, 95))
    tot_med, tot_p95 = float(np.median(times_total)), float(np.percentile(times_total, 95))
    fps_estimado = 1000.0 / tot_med if tot_med > 0 else 0.0

    model_size_kb = os.path.getsize(MODEL_WEIGHTS_PATH) / 1024.0

    print("\n" + "-" * 70)
    print(f"{'ETAPA':<22} | {'MEDIANA (ms)':<15} | {'P95 (ms)':<15}")
    print("-" * 70)
    print(f"{'Preprocesamiento (OpenCV)':<22} | {prep_med:<15.3f} | {prep_p95:<15.3f}")
    print(f"{'Inferencia CNN (PyTorch)':<22} | {cnn_med:<15.3f} | {cnn_p95:<15.3f}")
    print(f"{'Pipeline Total Cuadro':<22} | {tot_med:<15.3f} | {tot_p95:<15.3f}")
    print("-" * 70)
    print(f"Rendimiento Estimado : ~{fps_estimado:.1f} FPS")
    print(f"Tamaño del Modelo    : {model_size_kb:.2f} KB ({model_size_kb / 1024:.2f} MB)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()