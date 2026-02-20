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

# Parámetro configurable: Número de bloques
N = 70  # Cambia este valor para probar con diferentes N (debe ser par)

#####FUNCTION FOR GENERATING INITIAL AND GOAL CONFIGURATIONS#####
def generate_configurations(N: int) -> tuple:
    """
    Genera las configuraciones inicial y objetivo según las reglas del problema.
    Si N es impar, la primera pila tendrá un bloque más que la segunda.
    """
    if N < 1:
        raise ValueError("N debe ser al menos 1")
    
    # Bloques alfabéticos
    blocks = [chr(ord('A') + i) for i in range(N)]
    
    # Configuración inicial: distribuir bloques entre las dos primeras pilas
    # Si N es par: N/2 en cada pila
    # Si N es impar: (N+1)/2 en la primera pila, (N-1)/2 en la segunda
    first_stack_size = (N + 1) // 2  # Esto da el tamaño correcto para N par e impar
    second_stack_size = N // 2
    
    initial_state = [
        blocks[:first_stack_size], 
        blocks[first_stack_size:first_stack_size + second_stack_size], 
        []
    ]
    
    # Configuración objetivo: patrón intercalado inverso en pila 2 (stack vacío inicial)
    # Para N=4: ["A","B"], ["C","D"], [] -> [[], [], ["D","B","C","A"]]
    # Para N=5: ["A","B","C"], ["D","E"], [] -> [[], [], ["E","C","D","B","A"]]
    # Para N=6: ["A","B","C"], ["D","E","F"], [] -> [[], [], ["F","C","E","B","D","A"]]
    
    goal_blocks = []
    stack1_blocks = blocks[first_stack_size:first_stack_size + second_stack_size][::-1]  # Segunda pila invertida
    stack0_blocks = blocks[:first_stack_size][::-1]  # Primera pila invertida
    
    # Intercalar: empezar con el último de la segunda pila (si existe)
    # Luego alternar entre las dos pilas
    max_len = max(len(stack0_blocks), len(stack1_blocks))
    
    for i in range(max_len):
        # Primero de la segunda pila (si existe)
        if i < len(stack1_blocks):
            goal_blocks.append(stack1_blocks[i])
        # Luego de la primera pila
        if i < len(stack0_blocks):
            goal_blocks.append(stack0_blocks[i])
    
    goal_state = [[], [], goal_blocks]
    
    return initial_state, goal_state

#####FUNCTION FOR EXTRACTING MOVES VECTOR#####
def extract_moves_vector(response_text: str) -> list[list]:
    # Buscar la línea que contiene "moves = ["
    match = re.search(r'moves\s*=\s*\[.*\]', response_text, re.DOTALL)
    if not match:
        raise ValueError("❌ No se encontró 'moves = [' en la respuesta.")
    
    raw_block = match.group(0).split('=')[1].strip()
    
    # Remover líneas que empiecen con # (comentarios)
    lines = raw_block.split('\n')
    cleaned_lines = [line for line in lines if not line.strip().startswith('#')]
    cleaned_block = '\n'.join(cleaned_lines).strip()
    
    # Remover cualquier texto después del último ]
    end = cleaned_block.rfind(']')
    if end != -1:
        cleaned_block = cleaned_block[:end+1]
    
    try:
        moves = ast.literal_eval(cleaned_block)
    except Exception as e:
        raise ValueError(f"❌ Error al convertir el bloque a lista: {e}")
    
    if not (isinstance(moves, list) and all(isinstance(m, list) and len(m) == 3 for m in moves)):
        raise ValueError("❌ El contenido extraído no es una lista válida de movimientos.")
    
    return moves

#####FUNCTION FOR SIMULATING MOVES#####
def simulate_moves(initial_state: list, moves: list) -> list:
    """
    Simula la secuencia de movimientos en el BlocksWorld.
    """
    state = [stack.copy() for stack in initial_state]
    
    for move in moves:
        block, from_stack, to_stack = move
        
        # Verificar que el stack origen no esté vacío
        if not state[from_stack]:
            raise ValueError(f"❌ Stack {from_stack} está vacío, no se puede mover {block}")
        
        # Verificar que el bloque en la cima coincida
        if state[from_stack][-1] != block:
            raise ValueError(f"❌ El bloque {block} no está en la cima del stack {from_stack}")
        
        # Verificar que los índices de stack sean válidos
        if from_stack < 0 or from_stack >= len(state) or to_stack < 0 or to_stack >= len(state):
            raise ValueError(f"❌ Índices de stack inválidos: {from_stack} -> {to_stack}")
        
        # Ejecutar movimiento
        moved_block = state[from_stack].pop()
        state[to_stack].append(moved_block)
    
    return state

# Generar configuraciones
initial_state, goal_state = generate_configurations(N)

print(f"Configuración inicial (N={N}):")
for i, stack in enumerate(initial_state):
    print(f"Stack {i}: {stack}")

print(f"\nConfiguración objetivo:")
for i, stack in enumerate(goal_state):
    print(f"Stack {i}: {stack}")

# System prompt
system_instruction = """
You are a helpful assistant. Solve this puzzle for me. In this puzzle, there are stacks of blocks, and the goal is to rearrange them into a target configuration using a sequence of moves where: • Only the topmost block from any stack can be moved. • Ablock can be placed either on an empty position or on top of another block. Example: With initial state [["A", "B"], ["C"], []] and goal state [["A"], ["B"], ["C"]], a solution might be: moves = [["C", 1, 2], ["B", 0, 1]] This means: Move block C from stack 1 to stack 2, then move block B from stack 0 to stack 1. Requirements: • When exploring potential solutions in your thinking process, always include the corresponding complete list of moves. • Ensure your final answer also includes the complete list of moves for final solution in the format: moves = [[block, from stack, to stack], ...]

IMPORTANT: Your response must be ONLY the list in the exact format 'moves = [[...], [...], ...]' with no additional text, comments, explanations, or variations. Any output with comments, extra text, or different formats is invalid and will not be accepted.
"""

# User prompt
def format_state_for_prompt(state):
    lines = []
    for i, stack in enumerate(state):
        if stack:
            stack_str = " ".join(stack) + " (top)"
        else:
            stack_str = "(empty)"
        lines.append(f"Stack {i}: {stack_str}")
    return "\n".join(lines)

contents = f"""
I have a puzzle with {N} blocks.

Initial state:
{format_state_for_prompt(initial_state)}

Goal state:
{format_state_for_prompt(goal_state)}

Find the minimum sequence of moves to transform the initial state into the goal state. Remember that only the topmost block of each stack can be moved.
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
        print("\nThought summary:")
        print(part.text)
        print()
    else:
        print("Answer:")
        print(part.text)
        print()
        final_answer += part.text

# Validación
success = False

try:
    moves = extract_moves_vector(final_answer)
    print("Movimientos extraídos:", moves)
    
    # Simular movimientos usando la misma lógica
    final_state = simulate_moves(initial_state, moves)
    print("Estado final simulado:", final_state)
    
    # Verificar si se alcanzó el objetivo
    if final_state == goal_state:
        success = True
        print("🎯 ¡Configuración objetivo alcanzada!")
    else:
        print("❌ El estado final no coincide con el objetivo.")
        
except ValueError as e:
    print(f"❌ Se ha producido un error al procesar movimientos: {e}")
    print("🛑 El experimento se detiene aquí debido a un movimiento inválido.")

# Extraer uso de tokens
usage = response.usage_metadata
prompt_tokens = usage.prompt_token_count
output_tokens = usage.candidates_token_count
total_tokens = usage.total_token_count

# Guardar resultados en CSV
results_value = 'ok' if success else 'fail'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"N{N}_{timestamp}"

headers = ['Name', 'N', 'tokens_prompt', 'tokens_candidates', 'tokens_total', 'results']
row = [experiment_name, N, prompt_tokens, output_tokens, total_tokens, results_value]

os.makedirs("results", exist_ok=True)
csv_path = os.path.join("results", "blocks_world_baseline.csv")

file_exists = os.path.exists(csv_path)
with open(csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(headers)
    writer.writerow(row)

print(f"\n📄 Resultados guardados en: {csv_path}")
print(f"Resumen: {experiment_name} - Tokens totales: {total_tokens} - Resultado: {results_value}")
