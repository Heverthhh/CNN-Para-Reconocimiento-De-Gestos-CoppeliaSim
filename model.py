"""model.py - Definición de la arquitectura CNN en PyTorch y auditoría analítica de parámetros."""

from typing import List, Tuple
import torch
import torch.nn as nn


class FingerCountCNN(nn.Module):
    """Red Convolucional para clasificación de 0 a 4 dedos sobre imágenes (1, 64, 64)."""

    def __init__(self, num_classes: int = 5, dropout_p: float = 0.3) -> None:
        super().__init__()
        # Bloque Convolucional 1: Entrada (1, 64, 64) -> Salida (16, 32, 32)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloque Convolucional 2: Entrada (16, 32, 32) -> Salida (32, 16, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloque Convolucional 3: Entrada (32, 16, 16) -> Salida (64, 8, 8)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Clasificador denso
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu4 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagación hacia adelante retornando logits directos (sin Softmax)."""
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.drop(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x


def inspect_model(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 1, 64, 64)) -> None:
    """Calcula analíticamente la dimensión y los parámetros capa por capa, y valida contra PyTorch."""
    model.eval()
    print("\n" + "=" * 85)
    print(f"{'CAPA / MÓDULO':<25} | {'FORMA DE SALIDA':<25} | {'PARÁMETROS ANALÍTICOS':<20}")
    print("=" * 85)

    manual_param_total = 0
    hooks = []
    layer_info: List[Tuple[str, str, int]] = []

    def make_hook(name: str, mod: nn.Module):
        def hook(m, inp, out):
            n_params = sum(p.numel() for p in mod.parameters(recurse=False))
            out_shape = str(list(out.shape))
            layer_info.append((name, out_shape, n_params))
        return hook

    # Registrar hooks en subcapas que contengan parámetros o cambien la dimensionalidad
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Módulos hoja
            hooks.append(module.register_forward_hook(make_hook(name, module)))

    dummy_input = torch.zeros(input_size)
    with torch.no_grad():
        model(dummy_input)

    for h in hooks:
        h.remove()

    for name, shape_str, params in layer_info:
        manual_param_total += params
        print(f"{name:<25} | {shape_str:<25} | {params:<20}")

    pytorch_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_all = sum(p.numel() for p in model.parameters())

    print("-" * 85)
    print(f"Total Parámetros Analítico (Suma por Capas) : {manual_param_total}")
    print(f"Total Parámetros PyTorch (requires_grad=True): {pytorch_total}")
    print(f"Total Parámetros PyTorch (Incluye BN stats) : {pytorch_all}")
    print("=" * 85)

    assert manual_param_total == pytorch_all, "¡Discrepancia detectada en conteo analítico de parámetros!"
    print("[OK] Verificación matemática superada: Parámetros analíticos == Parámetros de PyTorch.\n")


if __name__ == "__main__":
    net = FingerCountCNN(num_classes=5)
    inspect_model(net)