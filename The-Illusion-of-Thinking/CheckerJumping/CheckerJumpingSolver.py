from google import genai
from google.genai import types
import os
import json
import re
import ast
import csv
from datetime import datetime

# Configura la API
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_HANOI"))

# Parámetro configurable: Número de checkers por color
N = 13  # Cambia este valor para probar con diferentes N (ej. 1, 3, etc.)

#####FUNCTION FOR EXTRACTING MOVES VECTOR#####
def extract_moves_vector(response_text: str) -> list[list]:
    # Buscar la línea que contiene "moves = ["
    match = re.search(r'moves\s*=\s*\[.*\]', response_text, re.DOTALL)
    if not match:
        raise ValueError("❌ No se encontró 'moves = [' en la respuesta.")
    
    raw_block = match.group(0).split('=')[1].strip()
    print(f"DEBUG: raw_block = {raw_block}")  # Debug
    
    # Remover líneas que empiecen con # (comentarios)
    lines = raw_block.split('\n')
    cleaned_lines = [line for line in lines if not line.strip().startswith('#')]
    cleaned_block = '\n'.join(cleaned_lines).strip()
    
    # Remover cualquier texto después del último ]
    end = cleaned_block.rfind(']')
    if end != -1:
        cleaned_block = cleaned_block[:end+1]
    
    print(f"DEBUG: cleaned_block = {cleaned_block}")  # Debug
    
    try:
        moves = ast.literal_eval(cleaned_block)
    except Exception as e:
        raise ValueError(f"❌ Error al convertir el bloque a lista: {e}")
    
    if not (isinstance(moves, list) and all(isinstance(m, list) and len(m) == 3 for m in moves)):
        raise ValueError("❌ El contenido extraído no es una lista válida de movimientos.")
    
    return moves

#####FUNCTION FOR SIMULATING MOVES#####
def simulate_moves(initial_board: list, moves: list) -> list:
    board = initial_board.copy()
    for move in moves:
        color, from_pos, to_pos = move
        if board[from_pos] != color:
            raise ValueError(f"❌ Movimiento inválido: {color} no está en posición {from_pos}")
        if board[to_pos] != '_':
            raise ValueError(f"❌ Posición destino {to_pos} no está vacía")
        
        # Verificar si es slide o jump
        if abs(from_pos - to_pos) == 1:
            # Slide
            pass
        elif abs(from_pos - to_pos) == 2:
            # Jump: verificar que hay una pieza opuesta en medio
            mid_pos = (from_pos + to_pos) // 2
            opposite = 'B' if color == 'R' else 'R'
            if board[mid_pos] != opposite:
                raise ValueError(f"❌ Salto inválido: no hay {opposite} en posición {mid_pos}")
        else:
            raise ValueError(f"❌ Movimiento inválido: distancia {abs(from_pos - to_pos)} no permitida")
        
        # Ejecutar movimiento
        board[from_pos] = '_'
        board[to_pos] = color
    
    return board

# System prompt
system_instruction = """
You are a helpful assistant. Solve this puzzle for me. On a one-dimensional board, there are red checkers (’R’), blue checkers (’B’), and one empty space (’_’). A checker can move by either: 1. Sliding forward into an adjacent empty space, or 2. Jumping over exactly one checker of the opposite color to land in an empty space. The goal is to swap the positions of all red and blue checkers, effectively mirroring the initial state. Example: If the initial state is [’R’, ’_’, ’B’], the goal is to reach [’B’, ’_’, ’R’]. Your solution should be a list of moves where each move is represented as [checker_color, position_from, position_to]. For example: moves = [[’R’, 0, 1], [’B’, 2, 0], [’R’, 1, 2]] position 2 to 0, and so on. Requirements: This means: Move the red checker from position 0 to 1, then move the blue checker from • When exploring potential solutions in your thinking process, always include the corresponding complete list of moves. • The positions are 0-indexed (the leftmost position is 0). • Ensure your final answer includes the complete list of moves for final solution in the format: moves = [[checker_color, position_from, position_to], ...]

IMPORTANT: Your response must be ONLY the list in the exact format 'moves = [[...], [...], ...]' with no additional text, comments, explanations, or variations. Any output with comments, extra text, or different formats is invalid and will not be accepted.
"""

# User prompt
contents = f"""
I have a puzzle with 2${N}$+1 positions, where ${N}$ red checkers (’R’) on left, ${N}$ blue checkers (’B’) on right, and one empty space (’_’) in between are arranged in a line. Initial board: {' '.join(['R'] * N + ['_'] + ['B'] * N)} Goal board: {' '.join(['B'] * N + ['_'] + ['R'] * N)} Rules: • Achecker can slide into an adjacent empty space. • Achecker can jump over exactly one checker of the opposite color to land in an empty space. • Checkers cannot move backwards (towards their starting side). Find the minimum sequence of moves to transform the initial board into the goal board.
"""

response = client.models.generate_content(
    model="gemini-2.5-pro-preview-06-05",
    config=types.GenerateContentConfig(
        system_instruction=system_instruction,
        thinking_config=types.ThinkingConfig(include_thoughts=True)
    ),
    contents=contents
)

# Procesar respuesta
final_answer = ""
for part in response.candidates[0].content.parts:
    if not part.text:
        continue
    if part.thought:
        print("Thought summary:")
        print(part.text)
        print()
    else:
        print("Answer:")
        print(part.text)
        print()
        final_answer += part.text

# Validación
initial_board = ['R'] * N + ['_'] + ['B'] * N
goal_board = ['B'] * N + ['_'] + ['R'] * N
success = False

try:
    moves = extract_moves_vector(final_answer)
    print("Movimientos extraídos:", moves)
    
    final_board = simulate_moves(initial_board, moves)
    print("Tablero final:", final_board)
    
    if final_board == goal_board:
        success = True
        print("🎯 ¡Objetivo alcanzado!")
    else:
        print("❌ El tablero final no coincide con el objetivo.")
        
except ValueError as e:
    print(f"❌ Error al procesar movimientos: {e}")

# Tokens
usage = response.usage_metadata
prompt_tokens = usage.prompt_token_count
output_tokens = usage.candidates_token_count
total_tokens = usage.total_token_count

# Guardar en CSV
results_value = 'ok' if success else 'fail'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"N{N}_{timestamp}"

headers = ['Name', 'N', 'tokens_prompt', 'tokens_candidates', 'tokens_total', 'results']
row = [experiment_name, N, prompt_tokens, output_tokens, total_tokens, results_value]

os.makedirs("results", exist_ok=True)
csv_path = os.path.join("results", "checker_jumping_baseline.csv")

file_exists = os.path.exists(csv_path)
with open(csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(headers)
    writer.writerow(row)

print(f"\n📄 Resultados guardados en: {csv_path}")
print(f"Resumen: {experiment_name} - Tokens totales: {total_tokens} - Resultado: {results_value}")
