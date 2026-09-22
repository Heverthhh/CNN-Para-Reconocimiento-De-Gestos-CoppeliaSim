"""test_adapter.py - Validación directa de CoppeliaSimAdapter (Secuencia 1 -> 2 -> 3 -> 4)."""

import time
import yaml
from robot_adapter import CoppeliaSimAdapter, MockAdapter

CONFIG_PATH = "config.yaml"


def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("\n" + "=" * 65)
    print(" INICIANDO PRUEBA FUNCIONAL DEL ADAPTADOR ROBÓTICO")
    print("=" * 65)

    adapter = CoppeliaSimAdapter(cfg)
    if not adapter.sim:
        print("[AVISO] CoppeliaSim no detectado. Utilizando MockAdapter para prueba en seco.\n")
        adapter = MockAdapter()

    try:
        # 1. Secuencia de prueba: Mover Articulación 1, 2, 3 y Pinza
        for cmd in [1, 2, 3, 4]:
            print(f"\n--- Probando Comando: {cmd} ---")
            res = adapter.execute(cmd)
            print(f"Envío de comando: {res}")

            # Prueba de rechazo inmediato si se envía un comando mientras está ocupado
            time.sleep(0.05)
            if adapter.is_busy():
                res_interferente = adapter.execute(cmd)
                print(f"Prueba de concurrencia (debe ser rechazado): {res_interferente}")

            # Esperar a que el robot concluya su trayectoria
            while adapter.is_busy():
                time.sleep(0.05)

            st = adapter.status()
            print(f"Estado tras movimiento: {st}")
            time.sleep(0.5)

        # 2. Alternar la pinza nuevamente para cerrarla / abrirla
        print("\n--- Probando alternancia de Pinza (Comando 4) ---")
        adapter.execute(4)
        while adapter.is_busy():
            time.sleep(0.05)
        print(f"Estado de pinza: {adapter.status()}")

        # 3. Probar Parada de Emergencia (Comando 0)
        print("\n--- Probando Comando de Parada (0) ---")
        stop_res = adapter.execute(0)
        print(f"Resultado parada: {stop_res}")

        # 4. Restablecer escena a pose inicial
        if hasattr(adapter, "reset_scene"):
            print("\n--- Restableciendo Escena ---")
            adapter.reset_scene()

        print("\n" + "=" * 65)
        print(" [OK] PRUEBA DE ADAPTADOR FINALIZADA EXITOSAMENTE")
        print("=" * 65 + "\n")

    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()