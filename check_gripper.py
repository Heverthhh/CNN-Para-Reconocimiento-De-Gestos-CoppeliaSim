"""check_gripper.py - Diagnostica el motor de la pinza PGripStraight."""
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require("sim")

print("\n--- BUSCANDO COMPONENTES DE LA PINZA ---")
# Buscar por alias directo
candidates = [
    "/PGripStraight",
    "/PGripStraight/motor",
    "/P_arm/PGripStraight",
    "/P_arm/PGripStraight/motor"
]

found = False
for c in candidates:
    try:
        h = sim.getObject(c)
        print(f"[ENCONTRADO] {c} -> Handle: {h}")
        found = True
    except Exception:
        pass

# Buscar cualquier objeto que contenga 'PGrip' o 'motor' en la escena
objs = sim.getObjectsInTree(sim.handle_scene)
for o in objs:
    alias = sim.getObjectAlias(o)
    if "pgrip" in alias.lower() or "motor" in alias.lower():
        print(f"[OBJETO EN ESCENA] Alias: '{alias}' | Handle: {o} | Tipo: {sim.getObjectType(o)}")

print("----------------------------------------\n")