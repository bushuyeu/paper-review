import os
import json
import requests
import re
import ast
from datetime import datetime
import csv

from HanoiTowersViewers import HanoiVisualizer  # se mantiene igual

# =========================
# Ollama (OpenAI-compatible) config
# =========================
# Endpoint OpenAI-compatible de Ollama:
LM_STUDIO_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/v1/chat/completions")
# Modelo DeepSeek-R1 14B en Ollama:
LM_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:14b")

# ========== HELPERS ==========
class SimpleUsage:
    """Normalize Ollama usage to your expected fields."""
    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=0):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = completion_tokens
        self.total_token_count = total_tokens

def parse_think_and_answer(text: str):
    """
    Split DeepSeek-R1 style <think>... </think> from the visible answer.
    Returns (thought_text_or_None, answer_text).
    """
    if not text:
        return None, ""
    # Robust parse of first <think>...</think> block
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        thought = m.group(1).strip()
        answer = (text[:m.start()] + text[m.end():]).strip()
        return thought, answer
    return None, text.strip()

# =========================
# PROMPT BUILDING (igual)
# =========================
def build_hanoi_prompt(N: int, k: list[list[int]], p: int) -> str:
    def format_peg(peg_index: int, peg: list[int]) -> str:
        if not peg:
            return f"    • Peg {peg_index}: (empty)"
        elif len(peg) == 1:
            return f"    • Peg {peg_index}: {peg[0]} (top)"
        else:
            return (
                f"    • Peg {peg_index}: {peg[0]} (bottom),"
                + ",".join(str(d) for d in peg[1:-1])
                + f",{peg[-1]} (top)"
            )

    peg_descriptions = "\n".join(format_peg(i, peg) for i, peg in enumerate(k))
    goal_list = list(range(N, 0, -1))
    goal_str = (
        "    • Peg 0: (empty)\n"
        "    • Peg 1: (empty)\n"
        "    • Peg 2: $" + f"{goal_list[0]}$ (bottom), ..." + f" {goal_list[-1]} (top)"
    )

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

# =========================
# ASK AGENT (Ollama)
# =========================
SYSTEM_INSTRUCTION = """
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
• The positions are 0-indexed (the leftmost peg is 0).
• Ensure your final answer includes the complete list of moves in the format: moves = [[disk id, from peg, to peg], ...]
• Dont overthink, do a quick analysis and the provide the output with the list of moves. Long thinking is penalyzed

The current configuration of the problem k, since it may be in the initial state or in an intermediate state.
• The current configuration of the problem k, since it may be in an intermediate state.
• The desired number of moves p. This parameter indicates how many moves I want you to make to bring us closer to the solution. This is because when the number of disks N is very large, solving the entire problem becomes very complex. Therefore, I don't want you to provide the complete solution, but rather the next p moves that move us toward the goal.

Your response should be just a vector of moves, without any additional text or explanations.
"""

def ask_hanoi_agent(contents: str):
    """
    Calls Ollama OpenAI-compatible chat completions with the same prompt structure.
    Prints a 'Thought summary' if a <think> block is present, then prints 'Answer'.
    Returns (final_answer_text, usage_like_object).
    """
    payload = {
        "model": LM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": contents},
        ],
        "temperature": 0.0,
        "max_tokens": 8192,  # ajusta según tu Ollama
    }

    resp = requests.post(LM_STUDIO_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()

    # Content (single choice)
    choice = data["choices"][0]
    content = choice.get("message", {}).get("content", "") or ""
    finish_reason = choice.get("finish_reason")

    # Print think + answer (emulación de Gemini thoughts)
    thought, answer = parse_think_and_answer(content)
    if thought:
        print("Thought summary:")
        print(thought)
        print()
    print("Answer:")
    print(answer)
    print()

    # Usage normalization (Ollama puede no devolver usage; caemos a 0)
    u = data.get("usage", {}) or {}
    usage = SimpleUsage(
        prompt_tokens=u.get("prompt_tokens", 0),
        completion_tokens=u.get("completion_tokens", 0),
        total_tokens=u.get("total_tokens", 0),
    )

    # Si se cortó por longitud, intenta continuar automáticamente una vez.
    if finish_reason == "length":
        cont_payload = {
            "model": LM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": contents},
                {"role": "assistant", "content": content},
                {"role": "user", "content": "Continue. Do not repeat any previous text. Just continue."},
            ],
            "temperature": 0.0,
            "max_tokens": 16184,  # ajusta según tu Ollama
        }
        cont = requests.post(LM_STUDIO_URL, json=cont_payload, timeout=600)
        cont.raise_for_status()
        cdata = cont.json()
        cchoice = cdata["choices"][0]
        ctext = cchoice.get("message", {}).get("content", "") or ""
        _, canswer = parse_think_and_answer(ctext)
        if canswer:
            answer += ("" if answer.endswith("\n") else "\n") + canswer
        cu = cdata.get("usage", {}) or {}
        usage = SimpleUsage(
            prompt_tokens=usage.prompt_token_count + cu.get("prompt_tokens", 0),
            completion_tokens=usage.candidates_token_count + cu.get("completion_tokens", 0),
            total_tokens=usage.total_token_count + cu.get("total_tokens", 0),
        )

    return answer, usage

# =========================
# EXTRACT MOVES (igual)
# =========================
def extract_moves_vector(response_text: str) -> list[list[int]]:
    """
    Extracts the first [[...]] list of integer triplets from a noisy LLM output.
    """
    start = response_text.find('[')
    end = response_text.rfind(']')

    if start == -1 or end == -1 or end <= start:
        raise ValueError("❌ No valid block delimited by [ and ] was found.")

    raw_block = response_text[start:end+1]
    cleaned_block = re.sub(r"[^\d\[\],-]", "", raw_block)  # permite posibles negativos si el modelo se equivoca

    try:
        moves = ast.literal_eval(cleaned_block)
    except Exception as e:
        raise ValueError(f"❌ Failed to convert cleaned block to list: {e}")
    
    if not (isinstance(moves, list) and all(isinstance(m, list) and len(m) == 3 for m in moves)):
        raise ValueError("❌ Extracted content is not a valid list of [disk, from, to] moves.")
    return moves

# =========================
# MAIN EXPERIMENT (idéntico)
# =========================
if __name__ == "__main__":
    N = 9  # Number of disks
    p = 150 # Number of moves per iteration

    k_init = [list(range(N, 0, -1)), [], []]
    goal_config = [[], [], list(range(N, 0, -1))]

    k_current = [peg[:] for peg in k_init]
    total_moves = []
    iteration = 0

    prompt_tokens = []
    output_tokens = []
    total_tokens = []
    success = False

    while True:
        iteration += 1
        print(f"\n🔄 Iteración {iteration} | Discos = {N} | p = {p}")
        try:
            # Build prompt
            prompt = build_hanoi_prompt(N=N, k=k_current, p=p)

            # Ask local LLM (Ollama)
            response_text, usage = ask_hanoi_agent(prompt)
            prompt_tokens.append(usage.prompt_token_count)
            output_tokens.append(usage.candidates_token_count)
            total_tokens.append(usage.total_token_count)

            # Extract moves
            moves = extract_moves_vector(response_text)

            # Accumulate
            total_moves.extend(moves)

            # Apply moves
            new_config = HanoiVisualizer.simulate_moves(k_current, moves)

            # Check goal
            if new_config == goal_config:
                success = True
                print("🎯 ¡Configuración objetivo alcanzada!")
                break

            # Next iteration
            k_current = new_config

        except ValueError as e:
            print(f"❌ Se ha producido un error en la iteración {iteration}: {e}")
            print("🛑 El experimento se detiene aquí debido a un movimiento inválido.")
            break

    # === FINAL REPORT ===
    print("\n✅ Secuencia de movimientos obtenida:" + str(total_moves))
    print("\n🎥 Visualizando secuencia completa de movimientos...")
    viz = HanoiVisualizer(k_init, total_moves)
    # viz.animate()

    results_value = 'ok' if success else 'fail'

    # Save CSV (idéntico)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"N{N}_p{p}_{timestamp}"

    max_iters = 10
    prompt_tokens += [''] * (max_iters - len(prompt_tokens))
    output_tokens += [''] * (max_iters - len(output_tokens))
    total_tokens += [''] * (max_iters - len(total_tokens))

    prompt_sum = sum([t for t in prompt_tokens if isinstance(t, int)])
    output_sum = sum([t for t in output_tokens if isinstance(t, int)])
    total_sum = sum([t for t in total_tokens if isinstance(t, int)])

    headers = ['Name'] + \
              [f"tokens_prompt_iter{i+1}" for i in range(max_iters)] + \
              [f"tokens_candidates_iter{i+1}" for i in range(max_iters)] + \
              [f"tokens_total_iter{i+1}" for i in range(max_iters)] + \
              ['tokens_prompt_sum', 'tokens_candidates_sum', 'tokens_total_sum','results']

    row = [experiment_name] + prompt_tokens + output_tokens + total_tokens + [prompt_sum, output_sum, total_sum, results_value]

    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "Deep_Seek_Steps_hanoi_token_usage.csv")

    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)

    print(f"\n📄 Resultados guardados en: {csv_path}")
