"""cnn_inference.py - Clasificador gestual en tiempo real con telemetría de latencia desacoplada."""

from pathlib import Path
import time
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model import FingerCountCNN
from preprocess import preprocess

CONFIG_PATH = "config.yaml"
MODELS_DIR = Path("models")


class GestureClassifier:
    """Carga el modelo entrenado y su configuración para realizar inferencia cuadro a cuadro."""

    def __init__(self, config_path: str = CONFIG_PATH) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights_path = MODELS_DIR / "best_model.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {weights_path.resolve()}")

        self.model = FingerCountCNN(num_classes=5).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.eval()

        # Warm-up inicial
        dummy = torch.zeros((1, 1, self.cfg["img_size"], self.cfg["img_size"]), device=self.device)
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(dummy)

    def predict(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Aplica preprocesamiento e inferencia midiendo tiempos desglosados."""
        t0 = time.perf_counter()

        # 1. Preprocesamiento común con los parámetros exactos de config.yaml
        x_np, hand_present, debug = preprocess(frame_bgr, self.cfg)
        t1 = time.perf_counter()

        # 2. Inferencia PyTorch
        x_tensor = torch.from_numpy(x_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        t2 = time.perf_counter()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        lat_prep = (t1 - t0) * 1000.0
        lat_cnn = (t2 - t1) * 1000.0
        lat_total = (t2 - t0) * 1000.0

        return {
            "clase": pred_class,
            "confianza": confidence,
            "probs": probs.tolist(),
            "hand_present": hand_present,
            "latencia_ms": {
                "preproc": lat_prep,
                "cnn": lat_cnn,
                "total": lat_total,
            },
            "debug": debug,
        }