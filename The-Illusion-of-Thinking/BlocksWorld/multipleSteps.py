import subprocess
import os

# Cambiar al directorio del script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Ejecutar BlocksWorldSolver.py 10 veces
for i in range(10):
    print(f"Ejecución {i+1}/10")
    try:
        subprocess.run(["python3", "BlocksWorldSolverSteps.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error en ejecución {i+1}: {e}")
    print()
