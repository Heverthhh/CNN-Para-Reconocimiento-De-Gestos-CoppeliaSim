"""test_live_view.py - Muestra los números y la imagen exacta que entra a la CNN."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from model import FingerCountCNN
from preprocess import preprocess

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FingerCountCNN(num_classes=5).to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device, weights_only=True))
model.eval()

cap = cv2.VideoCapture(int(cfg.get("camera_id", 0)))
print("\nPresiona [m] para alternar el modo ('mask' vs 'masked_gray' vs 'gray') en vivo!")
print("Presiona [q] para salir.\n")

modes = ["mask", "masked_gray", "gray"]
m_idx = modes.index(cfg.get("mode", "mask")) if cfg.get("mode") in modes else 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cfg["mode"] = modes[m_idx]
    x_np, present, debug = preprocess(frame, cfg)
    x_t = torch.from_numpy(x_np).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    # Imprimir probabilidades en vivo
    prob_str = " | ".join([f"C{i}: {probs[i]*100:4.1f}%" for i in range(5)])
    print(f"\rModo: {cfg['mode']:<11} | Mano: {str(present):<5} | {prob_str}", end="")

    # Ventana con lo que realmente "ve" la red
    cv2.imshow("Lo que ve la CNN (64x64 ampliado)", cv2.resize(debug["prep"], (256, 256), interpolation=cv2.INTER_NEAREST))

    k = cv2.waitKey(30) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('m'):
        m_idx = (m_idx + 1) % len(modes)

cap.release()
cv2.destroyAllWindows()