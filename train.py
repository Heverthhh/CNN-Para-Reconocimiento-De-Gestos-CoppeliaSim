"""train.py - Entrenamiento acelerado por GPU (CUDA) con Data Augmentation, caché y Early Stopping."""

import csv
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as T
import yaml

from model import FingerCountCNN
from preprocess import preprocess

# Rutas y Constantes del proyecto
CONFIG_PATH = "config.yaml"
MANIFEST_PATH = Path("data/manifest.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
SEED = 42

# Hiperparámetros de entrenamiento
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
PATIENCE = 7  # Épocas sin mejora en Balanced Accuracy antes de detener


def set_seed(seed: int = SEED) -> None:
    """Fija la semilla en todas las librerías para garantizar reproducibilidad exacta."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # En entrenamiento acelerado por GPU activamos benchmark para máxima velocidad
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Carga hiperparámetros desde config.yaml."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class CachedGestureDataset(Dataset):
    """Dataset que precalcula y mantiene en memoria RAM los tensores de preprocess()."""

    def __init__(
        self,
        records: List[Dict[str, str]],
        cfg: Dict[str, Any],
        transform: Any = None,
    ) -> None:
        self.transform = transform
        self.samples: List[Tuple[torch.Tensor, int]] = []

        sub_nombre = "TRAIN" if transform is not None else "VAL"
        print(f"Cachéando subconjunto [{sub_nombre}] ({len(records)} imágenes) con preprocess()...")

        for r in records:
            img_bgr = cv2.imread(r["ruta"])
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            local_cfg = cfg.copy()
            # Ajuste dinámico de ROI si la imagen es ya un recorte puro
            local_cfg["roi"] = [0, 0, w, h]

            # preprocess devuelve: x (1, S, S) float32 [0, 1]
            x, _, _ = preprocess(img_bgr, local_cfg)
            x_tensor = torch.from_numpy(x).float()  # Forma: (1, 64, 64)
            label = int(r["clase"])
            self.samples.append((x_tensor, label))

        print(f"-> Caché [{sub_nombre}] lista: {len(self.samples)} muestras preparadas.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x, y = self.samples[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def compute_balanced_accuracy(y_true: List[int], y_pred: List[int], num_classes: int = 5) -> float:
    """Calcula la exactitud balanceada: promedio de la exhaustividad (recall) por clase."""
    recalls = []
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    for c in range(num_classes):
        mask_c = (y_true_arr == c)
        total_c = np.sum(mask_c)
        if total_c > 0:
            correct_c = np.sum((y_pred_arr == c) & mask_c)
            recalls.append(correct_c / total_c)
        else:
            recalls.append(0.0)
    return float(np.mean(recalls))


def main() -> None:
    set_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.exists():
        print(f"Error: No existe {MANIFEST_PATH}. Ejecuta primero split_dataset.py.")
        sys.exit(1)

    cfg = load_config(CONFIG_PATH)

    # Selección y configuración del hardware acelerador
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print("\n" + "=" * 65)
        print(f" ACELERACIÓN POR HARDWARE: ACTIVADA (GPU: {gpu_name})")
        print(f" Memoria de Video (VRAM): {vram_total:.2f} GB")
        print("=" * 65)
        use_pin_memory = True
    else:
        device = torch.device("cpu")
        print("\n[AVISO] No se detectó CUDA. Ejecutando en CPU.")
        use_pin_memory = False

    # Leer manifest.csv omitiendo estrictamente el split 'test'
    train_records = []
    val_records = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["split"] == "train":
                train_records.append(r)
            elif r["split"] == "val":
                val_records.append(r)

    print(f"\nSubconjuntos cargados -> Train: {len(train_records)} | Val: {len(val_records)}")
    print("[REGLA METODOLÓGICA] Split 'test' preservado y protegido contra fuga de datos.")

    # Aumentos SOLO en train sobre tensores PyTorch
    train_transforms = T.Compose([
        T.RandomAffine(
            degrees=(-12, 12),
            translate=(0.08, 0.08),
            scale=(0.92, 1.08),
            fill=0,
        ),
    ])

    train_dataset = CachedGestureDataset(train_records, cfg, transform=train_transforms)
    val_dataset = CachedGestureDataset(val_records, cfg, transform=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=use_pin_memory,
    )

    # Inicializar red en la GPU
    model = FingerCountCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_bal_acc": [],
        "val_bal_acc": [],
    }

    best_val_bal_acc = 0.0
    epochs_no_improve = 0
    best_model_path = MODELS_DIR / "best_model.pth"

    print("\n" + "=" * 70)
    print(" INICIANDO ENTRENAMIENTO DE LA CNN (CRITERIO: BALANCED ACCURACY)")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        # ---------------- FASE DE ENTRENAMIENTO ----------------
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=use_pin_memory)
            y_batch = y_batch.to(device, non_blocking=use_pin_memory)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            train_preds.extend(preds)
            train_targets.extend(y_batch.cpu().tolist())

        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_bacc = compute_balanced_accuracy(train_targets, train_preds)

        # ---------------- FASE DE VALIDACIÓN ----------------
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=use_pin_memory)
                y_batch = y_batch.to(device, non_blocking=use_pin_memory)

                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * x_batch.size(0)

                preds = torch.argmax(logits, dim=1).cpu().tolist()
                val_preds.extend(preds)
                val_targets.extend(y_batch.cpu().tolist())

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_bacc = compute_balanced_accuracy(val_targets, val_preds)

        # Ajuste adaptativo del Learning Rate
        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(epoch_val_bacc)
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after < lr_before:
            print(f"  [SCHEDULER] Learning Rate reducido de {lr_before:.6f} a {lr_after:.6f}")

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_bal_acc"].append(epoch_train_bacc)
        history["val_bal_acc"].append(epoch_val_bacc)

        print(
            f"Época [{epoch:02d}/{EPOCHS}] | "
            f"Train Loss: {epoch_train_loss:.4f} - BAcc: {epoch_train_bacc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} - BAcc: {epoch_val_bacc*100:.2f}% | LR: {lr_after:.5f}"
        )

        # Comprobación de Early Stopping y guardado del mejor modelo
        if epoch_val_bacc > best_val_bal_acc:
            best_val_bal_acc = epoch_val_bacc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> [NUEVO MEJOR MODELO] Guardado checkpoint: {best_val_bal_acc*100:.2f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n[PARADA TEMPRANA] Sin mejora en {PATIENCE} épocas consecutivas.")
                break

    # Guardar metadatos del experimento
    meta_info = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "semilla": SEED,
        "dispositivo_utilizado": str(device),
        "gpu_modelo": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "modo_preprocesamiento": cfg.get("mode"),
        "img_size": cfg.get("img_size"),
        "cr": cfg.get("cr"),
        "cb": cfg.get("cb"),
        "min_area_frac": cfg.get("min_area_frac"),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "mejor_val_balanced_accuracy": best_val_bal_acc,
    }

    with open(MODELS_DIR / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=4)
    print(f"\n-> Metadatos guardados en: {(MODELS_DIR / 'training_meta.json').resolve()}")

    # Graficar curvas de entrenamiento y validación
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", color="blue")
    axes[0].plot(history["val_loss"], label="Val Loss", color="red")
    axes[0].set_title("Curva de Pérdida (Cross-Entropy)")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(history["train_bal_acc"], label="Train Bal Acc", color="blue")
    axes[1].plot(history["val_bal_acc"], label="Val Bal Acc", color="red")
    axes[1].set_title("Exactitud Balanceada")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Balanced Accuracy")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "training_curves.png", dpi=150)
    plt.close()
    print(f"-> Curvas guardadas en: {(REPORTS_DIR / 'training_curves.png').resolve()}\n")


if __name__ == "__main__":
    main()