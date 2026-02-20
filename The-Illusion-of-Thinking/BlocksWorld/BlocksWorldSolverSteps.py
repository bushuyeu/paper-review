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

# Parámetros configurables
N = 20  # Número de bloques (usar números pequeños para stepwise)
p = 25  # Número de pasos por iteración

#####FUNCTION FOR BUILDING THE PROMPT#####
def build_blocks_prompt(current_state: list, goal_state: list, N: int, p: int) -> str:
    """
    Construye el prompt para BlocksWorld con información del estado actual y objetivo.
    """
    def format_state_for_prompt(state):
        lines = []
        for i, stack in enumerate(state):
            if stack:
                stack_str = " ".join(stack) + " (top)"
            else:
                stack_str = "(empty)"
            lines.append(f"    • Stack {i}: {stack_str}")
        return "\n".join(lines)

    current_description = format_state_for_prompt(current_state)
    goal_description = format_state_for_prompt(goal_state)

    prompt = f"""
I have a puzzle with {N} blocks and I want to make {p} moves to bring us closer to the solution:

Current state:
{current_description}

Goal state:
{goal_description}

Rules:
• Only the topmost block from any stack can be moved.
• A block can be placed either on an empty position or on top of another block.

Find the next {p} moves to transform the current state closer to the goal state.
"""
    return prompt
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

#####FUNCTION FOR ASKING THE AGENT#####
def ask_blocks_agent(contents: str) -> tuple:
    """
    Interactúa con Gemini para resolver BlocksWorld step by step.
    """
    response = client.models.generate_content(
        model="gemini-2.5-pro-preview-06-05",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            thinking_config=types.ThinkingConfig(include_thoughts=True)
        ),
        contents=contents
    )

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

    return final_answer, response.usage_metadata
#####FUNCTION FOR EXTRACTING MOVES VECTOR#####
def extract_moves_vector(response_text: str) -> list[list]:
    """
    Extrae el vector de movimientos de la respuesta del LLM.
    Busca múltiples formatos posibles: moves = [...], JSON, o simplemente [...].
    """
    import json
    
    # Primero intentar el formato "moves = ["
    match = re.search(r'moves\s*=\s*\[.*\]', response_text, re.DOTALL)
    if match:
        raw_block = match.group(0).split('=')[1].strip()
    else:
        # Buscar formato JSON o simplemente un array
        # Buscar cualquier lista que contenga sublistas de 3 elementos
        start = response_text.find('[')
        end = response_text.rfind(']')
        
        if start == -1 or end == -1 or end <= start:
            raise ValueError("❌ No se encontró un bloque válido delimitado por [ y ].")
        
        raw_block = response_text[start:end+1]
    
    # Remover líneas que empiecen con # (comentarios)
    lines = raw_block.split('\n')
    cleaned_lines = [line for line in lines if not line.strip().startswith('#')]
    cleaned_block = '\n'.join(cleaned_lines).strip()
    
    # Remover cualquier texto después del último ]
    end = cleaned_block.rfind(']')
    if end != -1:
        cleaned_block = cleaned_block[:end+1]
    
    # Limpiar caracteres extra como comillas, etc.
    cleaned_block = re.sub(r'["`]', '', cleaned_block)
    
    try:
        # Primero intentar con ast.literal_eval
        moves = ast.literal_eval(cleaned_block)
    except:
        try:
            # Si falla, intentar con json
            moves = json.loads(cleaned_block)
        except:
            # Si todo falla, intentar parsing manual
            # Buscar patrones como ["D", 1, 2] o ['D', 1, 2]
            pattern = r'\[[\'"]*([A-Za-z]+)[\'"]*,\s*(\d+),\s*(\d+)\]'
            matches = re.findall(pattern, cleaned_block)
            
            if not matches:
                raise ValueError(f"❌ No se pudieron extraer movimientos del bloque: {cleaned_block}")
            
            moves = []
            for match in matches:
                block = match[0]  # Ya está limpio
                from_stack = int(match[1])
                to_stack = int(match[2])
                moves.append([block, from_stack, to_stack])
    
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

# System prompt modificado para stepwise reasoning
system_instruction = """
You are a helpful assistant. Solve this puzzle for me. In this puzzle, there are stacks of blocks, and the goal is to rearrange them into a target configuration using a sequence of moves where:
• Only the topmost block from any stack can be moved.
• A block can be placed either on an empty position or on top of another block.

Example: With initial state [["A", "B"], ["C"], []] and goal state [["A"], ["B"], ["C"]], a solution might be: moves = [["C", 1, 2], ["B", 0, 1]]
This means: Move block C from stack 1 to stack 2, then move block B from stack 0 to stack 1.

Requirements:
• When exploring potential solutions in your thinking process, always include the corresponding complete list of moves.
• Ensure your final answer includes the complete list of moves in the format: moves = [[block, from stack, to stack], ...]

The current configuration of the problem may be in an intermediate state.
• The desired number of moves p. This parameter indicates how many moves I want you to make to bring us closer to the solution. Since solving the entire problem at once can be complex, I don't want you to provide the complete solution, but rather the next p moves that move us toward the goal.

Your response should be just a vector of moves, without any additional text or explanations.
"""

######MAIN EXPERIMENT######
# Generar configuraciones
initial_state, goal_state = generate_configurations(N)

print(f"Configuración inicial (N={N}):")
for i, stack in enumerate(initial_state):
    print(f"Stack {i}: {stack}")

print(f"\nConfiguración objetivo:")
for i, stack in enumerate(goal_state):
    print(f"Stack {i}: {stack}")

# Inicializar variables para el bucle iterativo
current_state = [stack.copy() for stack in initial_state]
total_moves = []
iteration = 0

prompt_tokens = []
output_tokens = []
total_tokens = []
success = False

while True:
    iteration += 1
    print(f"\n🔄 Iteración {iteration} | Bloques = {N} | p = {p}")
    try:
        # Construir el prompt
        prompt = build_blocks_prompt(current_state, goal_state, N, p)

        # Preguntar al LLM
        response_text, usage = ask_blocks_agent(prompt)
        prompt_tokens.append(usage.prompt_token_count)
        output_tokens.append(usage.candidates_token_count)
        total_tokens.append(usage.total_token_count)

        # Extraer vector de movimientos
        moves = extract_moves_vector(response_text)
        print(f"🔍 Movimientos extraídos: {moves}")
        print(f"🔍 Primer movimiento: {moves[0] if moves else 'N/A'}")
        if moves:
            print(f"🔍 Tipo del primer elemento: {type(moves[0][0])}")
            print(f"🔍 Representación del primer elemento: {repr(moves[0][0])}")

        # Guardar movimientos acumulados
        total_moves.extend(moves)

        # Aplicar movimientos y obtener nueva configuración
        new_state = simulate_moves(current_state, moves)

        # Verificar si se alcanzó el objetivo
        if new_state == goal_state:
            success = True
            print("🎯 ¡Configuración objetivo alcanzada!")
            break

        # Preparar para siguiente iteración
        current_state = new_state

    except ValueError as e:
        print(f"❌ Se ha producido un error en la iteración {iteration}: {e}")
        print("🛑 El experimento se detiene aquí debido a un movimiento inválido.")
        break

# === REPORTE FINAL ===
print("\n✅ Secuencia de movimientos obtenida:" + str(total_moves))

# Guardar resultados en CSV (estilo steps)
results_value = 'ok' if success else 'fail'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"N{N}_p{p}_{timestamp}"

# Número máximo de iteraciones registrables
max_iters = 10

# Rellenar con valores vacíos si hay menos de 10 iteraciones
prompt_tokens += [''] * (max_iters - len(prompt_tokens))
output_tokens += [''] * (max_iters - len(output_tokens))
total_tokens += [''] * (max_iters - len(total_tokens))

# Sumas totales
prompt_sum = sum([t for t in prompt_tokens if isinstance(t, int)])
output_sum = sum([t for t in output_tokens if isinstance(t, int)])
total_sum = sum([t for t in total_tokens if isinstance(t, int)])

# Encabezado
headers = ['Name'] + \
          [f"tokens_prompt_iter{i+1}" for i in range(max_iters)] + \
          [f"tokens_candidates_iter{i+1}" for i in range(max_iters)] + \
          [f"tokens_total_iter{i+1}" for i in range(max_iters)] + \
          ['tokens_prompt_sum', 'tokens_candidates_sum', 'tokens_total_sum','results']

# Fila de datos
row = [experiment_name] + prompt_tokens + output_tokens + total_tokens + [prompt_sum, output_sum, total_sum, results_value]

# Guardar en CSV
os.makedirs("results", exist_ok=True)
csv_path = os.path.join("results", "blocks_world_steps.csv")

file_exists = os.path.exists(csv_path)
with open(csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(headers)
    writer.writerow(row)

print(f"\n📄 Resultados guardados en: {csv_path}")
print(f"Resumen: {experiment_name} - Tokens totales: {total_sum} - Resultado: {results_value}")
