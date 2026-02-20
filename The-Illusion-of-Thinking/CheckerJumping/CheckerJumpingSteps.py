from google import genai
from google.genai import types
import os
import json
from CheckerJumpingViewer import CheckerJumpingVisualizer
import re
import ast
import csv
from datetime import datetime

# Configura la API
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_HANOI"))

#####FUNCTION FOR BUILDING THE PROMPT#####
"""
This function builds a prompt for the Checker Jumping puzzle, including the current configuration of the board 
and the goal configuration. It formats the board in a readable way, ensuring that the checkers are described 
with their positions and colors. The prompt also includes the rules of the game and the number of moves.
"""
def build_checker_prompt(N: int, current_board: list, p: int) -> str:
    board_str = ' '.join(current_board)
    goal_board = ['B'] * N + ['_'] + ['R'] * N
    goal_str = ' '.join(goal_board)

    prompt = f"""
    I have a puzzle with 2${N}$+1 positions on a one-dimensional board with current configuration: {board_str}
    and I want to make ${p}$ moves to bring us closer to the solution.

    Current board: {board_str}
    Goal board: {goal_str}

    Rules:
    • A checker can slide into an adjacent empty space.
    • A checker can jump over exactly one checker of the opposite color to land in an empty space.
    • Checkers cannot move backwards (towards their starting side).
    • Red checkers ('R') can only move right, blue checkers ('B') can only move left.

    Find the next {p} moves to transform the current board configuration closer to the goal configuration.
    """
    return prompt

######FUNCTION FOR ASKING THE AGENT#####
"""
This function interacts with the Gemini AI model to solve the Checker Jumping puzzle.
It sends a prompt with the current configuration and the number of moves to make, and processes the response.
The response includes the thought process and the final answer, which is a list of moves to be made.
"""
def ask_checker_agent(contents: str) -> tuple:
    response = client.models.generate_content(
        model="gemini-2.5-pro-preview-06-05",
        config=types.GenerateContentConfig(
            system_instruction="""
            You are a helpful assistant. Solve this puzzle for me. On a one-dimensional board, there are red checkers ('R'), blue checkers ('B'), and one empty space ('_'). A checker can move by either: 1. Sliding forward into an adjacent empty space, or 2. Jumping over exactly one checker of the opposite color to land in an empty space. The goal is to swap the positions of all red and blue checkers, effectively mirroring the initial state.

            Your solution should be a list of moves where each move is represented as [checker_color, position_from, position_to]. For example: moves = [['R', 0, 1], ['B', 2, 0], ['R', 1, 2]]

            Requirements:
            • When exploring potential solutions in your thinking process, always include the corresponding complete list of moves.
            • The positions are 0-indexed (the leftmost position is 0).
            • Ensure your final answer includes the complete list of moves in the format: moves = [[checker_color, position_from, position_to], ...]
            • The current configuration of the problem, since it may be in the initial state or in an intermediate state.
            • The desired number of moves p. This parameter indicates how many moves I want you to make to bring us closer to the solution. Therefore, I don't want you to provide the complete solution, but rather the next p moves that move us toward the goal.

            IMPORTANT: Your response must be ONLY the list in the exact format 'moves = [[...], [...], ...]' with no additional text, comments, explanations, or variations. Any output with comments, extra text, or different formats is invalid and will not be accepted.
            """,
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
"""This function extracts the moves vector from the response text of the LLM.
It uses regular expressions to find the first occurrence of a list formatted as [[...]] and converts it to a Python list.
"""
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

######TESTING THE FUNCTION######
N = 6  # Number of checkers per color
p = 30  # Number of moves to make in each iteration

initial_board = ['R'] * N + ['_'] + ['B'] * N
goal_board = ['B'] * N + ['_'] + ['R'] * N

current_board = initial_board.copy()
total_moves = []
iteration = 0

prompt_tokens = []
output_tokens = []
total_tokens = []
success = False

while True:
    iteration += 1
    print(f"\n🔄 Iteración {iteration} | Checkers = {N} | p = {p}")
    try:
        # Construir el prompt
        prompt = build_checker_prompt(N=N, current_board=current_board, p=p)

        # Preguntar al LLM
        response_text, usage = ask_checker_agent(prompt)
        prompt_tokens.append(usage.prompt_token_count)
        output_tokens.append(usage.candidates_token_count)
        total_tokens.append(usage.total_token_count)

        # Extraer vector de movimientos
        moves = extract_moves_vector(response_text)

        # Guardar movimientos acumulados
        total_moves.extend(moves)

        # Aplicar movimientos y obtener nueva configuración
        states = CheckerJumpingVisualizer.simulate_moves(current_board, moves)
        if len(states) > len(moves):  # Si se pudieron aplicar todos los movimientos
            new_board = states[-1]
        else:
            print(f"❌ No se pudieron aplicar todos los movimientos. Estados: {len(states)}, Movimientos: {len(moves)}")
            break

        # Verificar si se alcanzó el objetivo
        if new_board == goal_board:
            success = True
            print("🎯 ¡Configuración objetivo alcanzada!")
            break

        # Preparar para siguiente iteración
        current_board = new_board
        print(f"Tablero actual: {' '.join(current_board)}")

    except ValueError as e:
        print(f"❌ Se ha producido un error en la iteración {iteration}: {e}")
        print("🛑 El experimento se detiene aquí debido a un movimiento inválido.")
        break

# === VISUALIZACIÓN FINAL ===
print(f"\n✅ Secuencia de movimientos obtenida ({len(total_moves)} movimientos):", total_moves)
print("\n🎥 Visualizando secuencia completa de movimientos...")
viz_states = CheckerJumpingVisualizer.simulate_moves(initial_board, total_moves)
print(f"Estados simulados: {len(viz_states)}")
for i, state in enumerate(viz_states):
    print(f"Paso {i}: {' '.join(state)}")

# Opcional: animar si hay movimientos válidos
if len(total_moves) > 0:
    # CheckerJumpingVisualizer.animate(initial_board, total_moves)  # Comentado para no mostrar visualizador
    pass

results_value = 'ok' if success else 'fail'

# Nombre del experimento
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
          ['tokens_prompt_sum', 'tokens_candidates_sum', 'tokens_total_sum', 'results']

# Fila de datos
row = [experiment_name] + prompt_tokens + output_tokens + total_tokens + [prompt_sum, output_sum, total_sum, results_value]

# Guardar en CSV
os.makedirs("results", exist_ok=True)
csv_path = os.path.join("results", "checker_jumping_steps.csv")

file_exists = os.path.exists(csv_path)
with open(csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(headers)
    writer.writerow(row)

print(f"\n📄 Resultados guardados en: {csv_path}")
