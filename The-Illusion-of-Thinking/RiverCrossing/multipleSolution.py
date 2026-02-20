import subprocess

for i in range(3):
    print(f"🔁 Ejecutando prueba {i + 1}/10")
    result = subprocess.run(["python3", "RiverCrossingSolver.py"], capture_output=True, text=True)
    print(result.stdout)  # Muestra la salida por consola (opcional)
    if result.stderr:
        print("⚠️ Error:", result.stderr)
