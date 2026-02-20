import subprocess

for i in range(10):
    print(f"🔁 Ejecutando prueba {i + 1}/10")
    result = subprocess.run(["python3", "HanoiTowersSolverConver.py"], capture_output=True, text=True)
    print(result.stdout)  # Muestra la salida por consola (opcional)
    if result.stderr:
        print("⚠️ Error:", result.stderr)
