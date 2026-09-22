"""run_ablation.py - Estudio sistemático de ablación comparando los 3 modos de preprocesamiento."""

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, List

import yaml

CONFIG_PATH = "config.yaml"
MODES = ["mask", "masked_gray", "gray"]
CSV_ABLATION_PATH = Path("reports/ablation.csv")


def run_command(cmd: List[str]) -> None:
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"Error ejecutando: {' '.join(cmd)}")
        sys.exit(res.returncode)


def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    original_mode = cfg.get("mode", "mask")
    results = []

    print("\n" + "=" * 70)
    print(" INICIANDO ESTUDIO DE ABLACIÓN (MODOS DE PREPROCESAMIENTO)")
    print(f" Modos a evaluar: {MODES}")
    print("=" * 70 + "\n")

    try:
        for mode in MODES:
            print(f"\n>>>> [ABLACIÓN] Entrenando con mode: '{mode}' <<<<")
            # 1. Modificar config.yaml
            cfg["mode"] = mode
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=None)

            # 2. Entrenar
            run_command([sys.executable, "train.py"])

            # 3. Guardar checkpoint específico
            ckpt_src = Path("models/best_model.pth")
            ckpt_dst = Path(f"models/best_model_{mode}.pth")
            shutil.copy(ckpt_src, ckpt_dst)

            # 4. Evaluar sobre conjunto Val
            run_command([
                sys.executable,
                "evaluate.py",
                "--modo",
                "val",
                "--model_path",
                str(ckpt_dst),
            ])

            # 5. Leer métricas de validación
            with open("reports/metrics_val.json", "r", encoding="utf-8") as f:
                metrics = json.load(f)

            bal_acc = metrics["balanced_accuracy"]
            f1_promedio = sum(m["f1_score"] for m in metrics["per_class"].values()) / len(metrics["per_class"])

            results.append({
                "modo": mode,
                "balanced_acc": f"{bal_acc * 100:.2f}%",
                "f1_promedio": f"{f1_promedio * 100:.2f}%",
                "c0_f1": f"{metrics['per_class']['0']['f1_score'] * 100:.1f}%",
                "c1_f1": f"{metrics['per_class']['1']['f1_score'] * 100:.1f}%",
                "c2_f1": f"{metrics['per_class']['2']['f1_score'] * 100:.1f}%",
                "c3_f1": f"{metrics['per_class']['3']['f1_score'] * 100:.1f}%",
                "c4_f1": f"{metrics['per_class']['4']['f1_score'] * 100:.1f}%",
            })

    finally:
        # Restaurar configuración original
        cfg["mode"] = original_mode
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=None)

    # Escribir tabla comparativa en reports/ablation.csv
    CSV_ABLATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["modo", "balanced_acc", "f1_promedio", "c0_f1", "c1_f1", "c2_f1", "c3_f1", "c4_f1"]
    with open(CSV_ABLATION_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("\n" + "=" * 70)
    print(" ESTUDIO DE ABLACIÓN COMPLETADO")
    print(f" Tabla comparativa guardada en: {CSV_ABLATION_PATH.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()