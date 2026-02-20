from google import genai
from google.genai import types
import os
import json # Sigue siendo útil para inspeccionar la respuesta completa si es necesario
from HanoiTowersViewers import HanoiVisualizer
import re
import ast
import csv
from datetime import datetime

# Configura la API
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_HANOI")) # Asegúrate de que la variable de entorno esté configurada

# Parámetro configurable: Número de discos
N = 10 # Cambia este valor para probar con diferentes N (ej. 3, 5, etc.)

#####FUNCTION FOR EXTRACTING MOVES VECTOR#####
"""This function extracts the moves vector from the response text of the LLM.
It uses regular expressions to find the first occurrence of a list formatted as [[...]] and converts it to a Python list.
"""

def extract_moves_vector(response_text: str) -> list[list[int]]:
    """
    Extrae el bloque de movimientos tipo [[1, 0, 2], ...] desde una salida ruidosa del LLM.
    Solo mantiene dígitos, comas y corchetes dentro del primer [ y el último ].

    Args:
        response_text (str): Texto completo devuelto por el modelo.

    Returns:
        list[list[int]]: Lista limpia de movimientos como objetos Python.
    """
    start = response_text.find('[')
    end = response_text.rfind(']')

    if start == -1 or end == -1 or end <= start:
        raise ValueError("❌ No se encontró un bloque válido delimitado por [ y ].")

    raw_block = response_text[start:end+1]
    cleaned_block = re.sub(r"[^\d\[\],]", "", raw_block)

    try:
        moves = ast.literal_eval(cleaned_block)
    except Exception as e:
        raise ValueError(f"❌ Error al convertir el bloque limpio a lista: {e}")
    
    if not (isinstance(moves, list) and all(isinstance(m, list) and len(m) == 3 and all(isinstance(x, int) for x in m) for m in moves)):
        raise ValueError("❌ El contenido extraído no es una lista válida de movimientos (listas de enteros).")

    return moves

response = client.models.generate_content(
    model="gemini-2.5-pro-preview-06-05", # O "gemini-2.5-flash-preview-06-05" para el modelo Flash
    config=types.GenerateContentConfig(
        system_instruction=f"""
        You are a helpful assistant. Solve this puzzle for me.
        There are three pegs and {N} disks of different sizes stacked on the first peg. The disks are numbered from 1 (smallest) to {N} (largest). Disk moves in this puzzle should follow:
        1. Only one disk can be moved at a time.
        2. Each move consists of taking the upper disk from one stack and placing it on top of another stack.
        3. A larger disk may not be placed on top of a smaller disk.

        The goal is to move the entire stack to the third peg.

        Example:
        With 3 disks numbered 1 (smallest), 2, and 3 (largest), the initial state is [[3, 2, 1], [], []], and a solution might be: moves = [[1, 0, 2], [2, 0, 1], [1, 2, 1], [3, 0, 2], [1, 1, 0], [2, 1, 2], [1, 0, 2]]
        This means: Move disk 1 from peg 0 to peg 2, then move disk 2 from peg 0 to peg 1, and so on.
        Requirements:
        • When exploring potential solutions in your thinking process, always include the corresponding complete list of moves.
        • The positions are 0-indexed (the leftmost peg is 0).
        • Ensure your final answer includes the complete list of moves in the format: moves = [[disk id, from peg, to peg], ...]

        Your response should be just a vector of moves, without any additional text or explanations.
    """,
        # ¡Esta es la forma correcta de solicitar las trazas de pensamiento!
        thinking_config=types.ThinkingConfig(
            include_thoughts=True
        )
    ),
    contents=f"""
    I have a puzzle with {N}$ disks of different sizes with Initial configuration:
    • Peg 0: ${N}$ (bottom), ... 2, 1 (top)
    • Peg 1: (empty)
    • Peg 2: (empty)

    Goal configuration:
    • Peg 0: (empty)
    • Peg 1: (empty)
    • Peg 2: ${N}$ (bottom), ... 2, 1 (top)

    Rules:
    • Only one disk can be moved at a time.
    • Only the top disk from any stack can be moved.
    • A larger disk may not be placed on top of a smaller disk. Find the sequence of moves to transform the initial configuration into the goal configuration.
    """
)

# Ahora, itera sobre las partes de la respuesta para imprimir el razonamiento y la respuesta.
final_answer = ""
for part in response.candidates[0].content.parts:
    if not part.text: # Salta partes vacías
        continue
    if part.thought:
        print("Thought summary:")
        print(part.text) # El texto asociado con el pensamiento
        print()
    else:
        print("Answer:")
        print(part.text) # La parte de la respuesta final
        print()
        final_answer += part.text  # Almacena respuesta útil

# Procesar y validar la respuesta
k_init = [list(range(N, 0, -1)), [], []]
goal_config = [[], [], list(range(N, 0, -1))]
success = False

try:
    moves = extract_moves_vector(final_answer)
    print("Movimientos extraídos:", moves)
    
    # Simular movimientos usando la misma lógica que HanoiTowersSolverSteps.py
    final_config = HanoiVisualizer.simulate_moves(k_init, moves)
    print("Configuración final simulada:", final_config)
    
    # Verificar si se alcanzó el objetivo (igual que en HanoiTowersSolverSteps.py)
    if final_config == goal_config:
        success = True
        print("🎯 ¡Configuración objetivo alcanzada!")
    else:
        print("❌ La configuración final no coincide con el objetivo.")
        
except ValueError as e:
    print(f"❌ Se ha producido un error al procesar movimientos: {e}")
    print("🛑 El experimento se detiene aquí debido a un movimiento inválido.")

# Extraer uso de tokens
usage = response.usage_metadata
prompt_tokens = usage.prompt_token_count
output_tokens = usage.candidates_token_count
total_tokens = usage.total_token_count

# Guardar resultados en CSV (como en HanoiTowersSolverSteps.py)
results_value = 'ok' if success else 'fail'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"N{N}_{timestamp}"

headers = ['Name', 'tokens_prompt', 'tokens_candidates', 'tokens_total', 'results']
row = [experiment_name, prompt_tokens, output_tokens, total_tokens, results_value]

os.makedirs("results", exist_ok=True)
csv_path = os.path.join("results", "baseline_Hanoi_Towers.csv")

file_exists = os.path.exists(csv_path)
with open(csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(headers)
    writer.writerow(row)

print(f"\n📄 Resultados guardados en: {csv_path}")
print(f"Resumen: {experiment_name} - Tokens totales: {total_tokens} - Resultado: {results_value}")