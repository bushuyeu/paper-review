import os
import google.generativeai as genai, types
import time
import re, json
from HanoiTowersViewers import HanoiVisualizer
from HanoiTowersViewers import HanoiVisualizer
import ast
from datetime import datetime
import csv



# Configura la API con tu clave
genai.configure(api_key=os.getenv("GEMINI_API_KEY_RIVER"))

# Inicializa los modelos generativos con instrucciones de sistema
model_a = genai.GenerativeModel(
    model_name="gemini-2.5-pro-preview-06-05",
    system_instruction="""
        You are a helpful assistant. Solve this puzzle for me.
        There are three pegs and n disks of different sizes stacked on the first peg. The disks are numbered from 1 (smallest) to n (largest). Disk moves in this puzzle should follow:
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

        Since the puzzle is complex when the number of disks N is very large, you will collaborate with another agent to solve it. Every time is your turn to interact, you will receive the following information:
        • The current configuration of the problem k, since it may be in the initial state or in an intermediate state.
        • The desired number of moves p. This parameter indicates how many moves I want you to make to bring us closer to the solution. This is because when the number of disks N is very large, solving the entire problem becomes very complex. Therefore, I don't want you to provide the complete solution, but rather the next p moves that move us toward the goal.


        Your response should be just a vector of moves, without any additional text or explanations. Example response:
        moves = [[1, 0, 2], [2, 0, 1], [1, 2, 1], [3, 0, 2], [1, 1, 0], [2, 1, 2]]
        """
)

model_b = genai.GenerativeModel(
    model_name="gemini-2.5-pro-preview-06-05",
    system_instruction="""
        You are a helpful assistant. Solve this puzzle for me.
        There are three pegs and n disks of different sizes stacked on the first peg. The disks are numbered from 1 (smallest) to n (largest). Disk moves in this puzzle should follow:
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

        Since the puzzle is complex when the number of disks N is very large, you will collaborate with another agent to solve it. Every time is your turn to interact, you will receive the following information:
        • The current configuration of the problem k, since it may be in the initial state or in an intermediate state.
        • The desired number of moves p. This parameter indicates how many moves I want you to make to bring us closer to the solution. This is because when the number of disks N is very large, solving the entire problem becomes very complex. Therefore, I don't want you to provide the complete solution, but rather the next p moves that move us toward the goal.


        Your response should be just a vector of moves, without any additional text or explanations. Example response:
        moves = [[1, 0, 2], [2, 0, 1], [1, 2, 1], [3, 0, 2], [1, 1, 0], [2, 1, 2]]
        """
)

# Inicia las sesiones de chat para ambos modelos
chat_a = model_a.start_chat(history=[])
chat_b = model_b.start_chat(history=[])

"""
This function builds a prompt for the Tower of Hanoi puzzle, including the current configuration of pegs and the goal configuration.
It formats the pegs and the goal in a readable way, ensuring that the disks are described with their positions and sizes.
The prompt also includes the rules of the game and the number of moves
"""
def build_hanoi_prompt(N: int, k: list[list[int]], p: int) -> str:
    def format_peg(peg_index: int, peg: list[int]) -> str:
        if not peg:
            return f"    • Peg {peg_index}: (empty)"
        elif len(peg) == 1:
            return f"    • Peg {peg_index}: {peg[0]} (top)"
        else:
            elements = ",".join(str(d) for d in peg[:-1])
            return f"    • Peg {peg_index}: {peg[0]} (bottom)," + ",".join(str(d) for d in peg[1:-1]) + f",{peg[-1]} (top)"

    peg_descriptions = "\n".join(format_peg(i, peg) for i, peg in enumerate(k))
    goal_list = list(range(N, 0, -1))
    goal_str = f"    • Peg 0: (empty)\n    • Peg 1: (empty)\n    • Peg 2: $" + f"{goal_list[0]}$ (bottom), ..." + f" {goal_list[-1]} (top)"

    prompt = f"""
    I have a puzzle with ${N}$ disks of different sizes with configuration k={k} and I want to make ${p}$ moves to bring us closer to the solution:
{peg_descriptions}

    Goal configuration k=[[],[],{goal_list}]:
{goal_str}

    Rules:
    • Only one disk can be moved at a time.
    • Only the top disk from any stack can be moved.
    • A larger disk may not be placed on top of a smaller disk. Find the sequence of moves to transform the initial configuration into the goal configuration.
    """
    return prompt


######FUNCTION FOR ASKING THE AGENT#####
"""
This function interacts with the Gemini AI model to solve the Tower of Hanoi puzzle.
It sends a prompt with the current configuration and the number of moves to make, and processes the response.
The response includes the thought process and the final answer, which is a list of moves to be made.
"""

def ask_hanoi_agent(contents: str) -> str:
    # Configura el cliente
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_HANOI"))

    # Solicita respuesta del modelo
    response = client.models.generate_content(
        model="gemini-2.5-pro-preview-06-05",
        config=types.GenerateContentConfig(
            system_instruction="""
        You are a helpful assistant. Solve this puzzle for me.
        There are three pegs and n disks of different sizes stacked on the first peg. The disks are numbered from 1 (smallest) to n (largest). Disk moves in this puzzle should follow:
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

        The current configuration of the problem k, since it may be in the initial state or in an intermediate state.
        • The current configuration of the problem k, since it may be in the initial state or in an intermediate state.
        • The desired number of moves p. This parameter indicates how many moves I want you to make to bring us closer to the solution. This is because when the number of disks N is very large, solving the entire problem becomes very complex. Therefore, I don't want you to provide the complete solution, but rather the next p moves that move us toward the goal.

        Your response should be just a vector of moves, without any additional text or explanations.
        """,
            thinking_config=types.ThinkingConfig(include_thoughts=True)
        ),
        contents=contents
    )

    # Procesamiento de respuesta
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
            final_answer += part.text  # Almacena respuesta útil

    return final_answer

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

########################################### Example usage ###################################################
def run_hanoi_experiment(N=4, p=10):
    # Parámetros iniciales
    # N = 4
    # p = 10
    k_actual = [list(range(N, 0, -1)), [], []]
    k_objetivo = [[], [], list(range(N, 0, -1))]
    total_moves = []
    turn = 0  # 0: chat_a, 1: chat_b

    # ─── 2. Variables de control ─────────────────────────────────────
    turn = 0                  # 0 -> A, 1 -> B
    last_resp_a = None
    last_resp_b = None

    # ─── 3. Primer mensaje SOLO al agente A (no hay colega previo) ───
    prompt_inicial = build_hanoi_prompt(N, k_actual, p)   # tu helper
    chat_a.send_message(prompt_inicial)

    prompt_tokens = []
    output_tokens = []
    total_tokens = []
    success = False

    # ─── 4. Bucle por turnos con paso de contexto ────────────────────
    while True:
        current_chat   = chat_a if turn == 0 else chat_b
        other_last_msg = last_resp_b if turn == 0 else last_resp_a
        agent_label    = "A" if turn == 0 else "B"

        # 4.1 Construir prompt completo
        if other_last_msg:
            prompt = (
                f"Last moves made by your colleague:\n{other_last_msg}\n\n"
                f"The current configuration is: {k_actual}\n"
                f"Please provide the next {p} moves to bring us closer to the solution.\n"
                f"Just answer with the list of moves."
            )
        else:  # primer turno de B
            prompt = (
                f"The current configuration is: {k_actual}\n"
                f"Please provide the next {p} moves to bring us closer to the solution.\n"
                f"Just answer with the list of moves."
            )

        # 4.2 Enviar y mostrar respuesta
        try:
            response = current_chat.send_message(prompt)
            print(f"\n🧠 Respuesta del modelo {agent_label}:\n{response.text}\n")

            # Extraer tokens
            usage = response.usage_metadata
            prompt_tokens.append(usage.prompt_token_count)
            output_tokens.append(usage.candidates_token_count)
            total_tokens.append(usage.total_token_count)

        except Exception as e:
            print("❌ Error al invocar el modelo:", e)
            break

        # 4.3 Guardar la respuesta del agente actual
        if turn == 0:
            last_resp_a = response.text
        else:
            last_resp_b = response.text

        # 4.4 Procesar la lista de movimientos y actualizar tablero
        try:
            moves = extract_moves_vector(response.text)       # tu helper
            k_actual = HanoiVisualizer.simulate_moves(k_actual, moves)
        except Exception as e:
            print("⚠️  Problema con los movimientos:", e)
            total_moves.extend(moves)
            print("✔️  Estado actualizado:", k_actual)
            break

        total_moves.extend(moves)
        print("✔️  Estado actualizado:", k_actual)

        if k_actual == k_objetivo:
            success = True
            print("🎯 ¡Objetivo alcanzado!")
            break

        # 4.5 Cambiar de turno
        turn = 1 - turn

    # Visualización final
    print(f"\n✅ Total de movimientos realizados: {len(total_moves)}")
    viz = HanoiVisualizer([[i for i in range(N, 0, -1)], [], []], total_moves)
    #viz.animate()

    # ─── GUARDAR RESULTADOS EN CSV ──────────────────────────────
    results_value = 'ok' if success else 'fail'
    max_iters = 10
    prompt_tokens += [''] * (max_iters - len(prompt_tokens))
    output_tokens += [''] * (max_iters - len(output_tokens))
    total_tokens += [''] * (max_iters - len(total_tokens))

    prompt_sum = sum([t for t in prompt_tokens if isinstance(t, int)])
    output_sum = sum([t for t in output_tokens if isinstance(t, int)])
    total_sum = sum([t for t in total_tokens if isinstance(t, int)])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"N{N}_p{p}_{timestamp}"

    headers = ['Name'] + \
            [f"tokens_prompt_iter{i+1}" for i in range(max_iters)] + \
            [f"tokens_candidates_iter{i+1}" for i in range(max_iters)] + \
            [f"tokens_total_iter{i+1}" for i in range(max_iters)] + \
            ['tokens_prompt_sum', 'tokens_candidates_sum', 'tokens_total_sum','results']

    row = [experiment_name] + prompt_tokens + output_tokens + total_tokens + [prompt_sum, output_sum, total_sum, results_value]


    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "hanoi_token_usage_conver.csv")

    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)

    print(f"\n📄 Resultados guardados en: {csv_path}")


# print("🚀 Iniciando experimento de la Torre de Hanoi con N=5 p=30...")
# for i in range(10):
#     print(f"\n🚀 Lanzando experimento {i+1}/10")
#     run_hanoi_experiment(N=5, p=30)

# print("🚀 Iniciando experimento de la Torre de Hanoi con N=7 p=60...")
# for i in range(5):
#     print(f"\n🚀 Lanzando experimento {i+1}/10")
#     run_hanoi_experiment(N=7, p=60)

print("🚀 Iniciando experimento de la Torre de Hanoi con N=8 p=100...")
for i in range(3):
    print(f"\n🚀 Lanzando experimento {i+1}/10")
    run_hanoi_experiment(N=8, p=100)

# print("🚀 Iniciando experimento de la Torre de Hanoi con N=9 p=150...")
# for i in range(5):
#     print(f"\n🚀 Lanzando experimento {i+1}/10")
#     run_hanoi_experiment(N=9, p=150)

# print("🚀 Iniciando experimento de la Torre de Hanoi con N=10 p=200...")
# for i in range(5):
#     print(f"\n🚀 Lanzando experimento {i+1}/10")
#     run_hanoi_experiment(N=10, p=200)