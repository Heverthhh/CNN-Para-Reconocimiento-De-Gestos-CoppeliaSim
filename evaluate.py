"""evaluate.py - Evaluación formal sobre splits Val o Test con auditoría de errores y salvaguarda."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model import FingerCountCNN
from preprocess import preprocess

CONFIG_PATH = "config.yaml"
MANIFEST_PATH = Path("data/manifest.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
CLASSES = [0, 1, 2, 3, 4]
CLASS_NAMES = ["0 (Parada)", "1 (Art 1)", "2 (Art 2)", "3 (Art 3)", "4 (Pinza)"]


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5):
    """Calcula la matriz de confusión, balanced accuracy y métricas por clase."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = {}
    recalls = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        sup = cm[c, :].sum()

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        recalls.append(rec)
        per_class[c] = {"precision": prec, "recall": rec, "f1_score": f1, "support": int(sup)}

    bal_acc = float(np.mean(recalls))
    return cm, bal_acc, per_class


def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path) -> None:
    """Grafica la matriz de confusión con conteos enteros y porcentajes normalizados por fila."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    with np.errstate(all="ignore"):
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    cax = ax.matshow(cm_norm, cmap=plt.cm.Blues, vmin=0, vmax=1)
    fig.colorbar(cax)

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="left")
    ax.set_yticklabels(CLASS_NAMES)

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            cnt = cm[i, j]
            pct = cm_norm[i, j] * 100.0
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cnt}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=9)

    ax.set_xlabel("Predicción CNN", fontweight="bold", labelpad=10)
    ax.set_ylabel("Clase Real (Ground Truth)", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_most_confident_errors(errors: List[Dict[str, Any]], out_path: Path, top_k: int = 10) -> None:
    """Genera un mosaico visual con los errores de predicción de mayor confianza."""
    if not errors:
        print("[INFO] No hubo errores en el split evaluado. Omitiendo mosaico de errores.")
        return

    # Ordenar errores por confianza descendente
    sorted_errors = sorted(errors, key=lambda e: e["confidence"], reverse=True)[:top_k]
    n_show = len(sorted_errors)
    cols = 5
    rows = int(np.ceil(n_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.2 * rows))
    axes_flat = axes.flatten() if rows > 1 else [axes] if n_show == 1 else axes

    for idx, err in enumerate(sorted_errors):
        ax = axes_flat[idx]
        img_bgr = cv2.imread(err["ruta"])
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
        ax.set_title(
            f"Real: C{err['true']} -> Pred: C{err['pred']}\nConf: {err['confidence']*100:.1f}%",
            fontsize=9,
            color="red",
        )
        ax.axis("off")

    for i in range(n_show, len(axes_flat)):
        axes_flat[i].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"-> Mosaico de errores más confiados guardado en: {out_path.resolve()}")


def evaluate_split(modo: str, model_weights: Path) -> None:
    """Ejecuta la inferencia sobre el split solicitado ('val' o 'test')."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(CONFIG_PATH)

    flag_file = REPORTS_DIR / "test_done.flag"
    if modo == "test" and flag_file.exists():
        print("\n" + "!" * 75)
        print("[ADVERTENCIA ESTRICTA] Ya existe 'reports/test_done.flag'.")
        print("El split 'test' debe evaluarse UNA SOLA VEZ para evitar fuga por sobreajuste manual.")
        print("!" * 75 + "\n")

    # Filtrar registros del manifiesto
    records = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["split"] == modo:
                records.append(r)

    if not records:
        print(f"Error: No hay registros con split='{modo}' en {MANIFEST_PATH}.")
        return

    print(f"\nEvaluando split: [{modo.upper()}] ({len(records)} imágenes)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FingerCountCNN(num_classes=5).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()

    y_true_list = []
    y_pred_list = []
    errors_list = []

    with torch.no_grad():
        for r in records:
            img_bgr = cv2.imread(r["ruta"])
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            local_cfg = cfg.copy()
            local_cfg["roi"] = [0, 0, w, h]

            x_np, _, _ = preprocess(img_bgr, local_cfg)
            x_tensor = torch.from_numpy(x_np).float().unsqueeze(0).to(device)

            logits = model(x_tensor)
            probs = F.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

            t_val = int(r["clase"])
            p_val = int(pred.item())
            c_val = float(conf.item())

            y_true_list.append(t_val)
            y_pred_list.append(p_val)

            if t_val != p_val:
                errors_list.append({
                    "ruta": r["ruta"],
                    "true": t_val,
                    "pred": p_val,
                    "confidence": c_val,
                })

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    cm, bal_acc, per_class = compute_metrics(y_true, y_pred, num_classes=5)

    # Gráfica de matriz de confusión
    cm_path = REPORTS_DIR / f"confusion_matrix_{modo}.png"
    plot_confusion_matrix(cm, f"Matriz de Confusión - Split {modo.upper()}", cm_path)

    # Formatear reporte en texto
    lines = []
    lines.append("=" * 70)
    lines.append(f" REPORTE DE EVALUACIÓN - SPLIT [{modo.upper()}]")
    lines.append("=" * 70)
    lines.append(f"Total de imágenes evaluadas : {len(y_true)}")
    lines.append(f"Exactitud Balanceada Global : {bal_acc * 100:.2f}%")
    lines.append("-" * 70)
    lines.append(f"{'Clase':<10} | {'Soporte':<8} | {'Precisión':<12} | {'Recall':<12} | {'F1-Score':<10}")
    lines.append("-" * 70)
    for c in CLASSES:
        m = per_class[c]
        lines.append(
            f"Clase {c:<4} | {m['support']:<8} | {m['precision']*100:>10.2f}% | "
            f"{m['recall']*100:>10.2f}% | {m['f1_score']*100:>8.2f}%"
        )
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    print("\n" + report_text + "\n")

    # Guardar métricas
    with open(REPORTS_DIR / f"evaluation_{modo}.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(REPORTS_DIR / f"metrics_{modo}.json", "w", encoding="utf-8") as f:
        json.dump(
            {"split": modo, "balanced_accuracy": bal_acc, "per_class": per_class, "confusion_matrix": cm.tolist()},
            f,
            indent=4,
        )

    # Acciones exclusivas del modo Test
    if modo == "test":
        plot_most_confident_errors(errors_list, REPORTS_DIR / "most_confident_errors_test.png")
        flag_file.touch()
        print(f"-> Creado archivo centinela: {flag_file.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación del modelo sobre splits Val o Test.")
    parser.add_argument("--modo", type=str, choices=["val", "test"], default="val", help="Split a evaluar.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/best_model.pth",
        help="Ruta al checkpoint del modelo.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: No existe el modelo en {model_path.resolve()}.")
        sys.exit(1)

    evaluate_split(args.modo, model_path)


if __name__ == "__main__":
    main()