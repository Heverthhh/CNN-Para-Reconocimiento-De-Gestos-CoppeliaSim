"""test_motor_mode.py - Prueba empírica de movimiento del motor de la pinza."""
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require("sim")

motor = sim.getObject("/PGripStraight/motor")

print("\n--- DIAGNÓSTICO EN VIVO DE LA PINZA ---")
pos_actual = sim.getJointPosition(motor)
print(f"Posición actual del motor: {pos_actual:.4f}")

# PRUEBA 1: Posición angular grande (0.8 radianes ~ 45 grados)
print("\n1. Probando control de posición angular (0.8 rad)...")
sim.setJointTargetPosition(motor, 0.8)
time.sleep(1.5)
print(f"Posición tras enviar 0.8 rad: {sim.getJointPosition(motor):.4f}")

# PRUEBA 2: Posición 0.0
print("\n2. Probando control de posición angular (0.0 rad)...")
sim.setJointTargetPosition(motor, 0.0)
time.sleep(1.5)
print(f"Posición tras enviar 0.0 rad: {sim.getJointPosition(motor):.4f}")

# PRUEBA 3: Control por velocidad (+0.05 m/s o rad/s)
print("\n3. Probando control por velocidad positiva (+0.05)...")
sim.setJointTargetVelocity(motor, 0.05)
time.sleep(1.5)
print(f"Posición tras velocidad positiva: {sim.getJointPosition(motor):.4f}")

# PRUEBA 4: Control por velocidad negativa (-0.05)
print("\n4. Probando control por velocidad negativa (-0.05)...")
sim.setJointTargetVelocity(motor, -0.05)
time.sleep(1.5)
print(f"Posición tras velocidad negativa: {sim.getJointPosition(motor):.4f}")

# Frenar
sim.setJointTargetVelocity(motor, 0.0)
print("\n---------------------------------------")