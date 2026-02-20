import os
import re
from typing import List
import requests
from datetime import datetime
import pandas as pd
from types import SimpleNamespace  # para mimetizar usage con atributos
from RiverCrossingViewer import RiverCrossingVisualizer

# =========================
# Ollama (OpenAI-compatible) config
# =========================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/v1/chat/completions")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:14b")


def build_river_crossing_prompt(N: int, k: int) -> str:
    """
    Build the river-crossing statement only when the instance is solvable.

    Solvability rules for the symmetric “N actors and N agents” variant
    (same logic as the missionaries-and-cannibals puzzle):

        • k == 1  →  impossible (no one can bring the boat back).
        • k == 2  →  solvable only if N ≤ 3  (upper bound 2k-1).
        • k == 3  →  solvable only if N ≤ 5  (upper bound 2k-1).
        • k ≥ 4   →  always solvable for any N ≥ 1.

    A ValueError is raised for any (N, k) pair that violates these rules.
    """
    if k < 1:
        raise ValueError("❌ Boat capacity k must be at least 1.")
    if N < 1:
        raise ValueError("❌ There must be at least one actor and one agent (N ≥ 1).")

    # Unsatisfiable cases
    if k == 1:
        raise ValueError("❌ A single-seat boat makes the puzzle impossible.")
    if k in (2, 3) and N > 2 * k - 1:
        raise ValueError(
            f"❌ The puzzle has no solution for N={N} and k={k} "
            f"(for k={k}, N must satisfy N ≤ {2 * k - 1})."
        )
    # For k ≥ 4 there is no upper bound on N

    return (
        f"{N} actors and their {N} agents want to cross a river in a boat that is capable of holding "
        f"only {k} people at a time, with the constraint that no actor can be in the presence of another agent, "
        f"including while riding the boat, unless their own agent is also present, because each agent is worried "
        f"their rivals will poach their client. Initially, all actors and agents are on the left side of the river "
        f"with the boat. How should they cross the river? (Note: the boat cannot travel empty)"
    )


def call_gemini_model(  # mantenemos el nombre y firma
    prompt_text: str,
    N: int,
    k: int,
    csv_path: str = "tokens_river.csv",
    model_name: str = "gemini-2.5-pro-preview-06-05"  # ignorado; se usa OLLAMA_MODEL
) -> str:
    # System prompt EXACTO (no modificado)
    system_instruction = (
        "You are a helpful assistant. Solve this puzzle for me. You can represent actors with a_1, a_2, ... "
        "and agents with A_1, A_2, ... . Your solution must be a list of boat moves where each move indicates "
        "the people on the boat. For example, if there were two actors and two agents, you should return: "
        "moves=[[\"A_2\", \"a_2\"], [\"A_2\"], [\"A_1\", \"A_2\"], [\"A_1\"], [\"A_1\", \"a_1\"]] which indicates "
        "that in the first move, A_2 and a_2 row from left to right, and in the second move, A_2 rows from right "
        "to left and so on. Requirements: • When exploring potential solutions in your thinking process, always "
        "include the corresponding complete list of boat moves. • The list shouldn’t have comments. • Ensure your "
        "final answer also includes the complete list of moves for final solution."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_text},
        ],
        # max_tokens/temperatura vía parámetros OpenAI-compatibles:
        "temperature": 0.0,
        "max_tokens": 8192
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()

    # Texto de salida (idéntico uso)
    content = data["choices"][0]["message"]["content"]

    # Normalizar 'usage' (si Ollama no lo devuelve, usamos 0s)
    u = data.get("usage") or {}
    usage = SimpleNamespace(
        prompt_token_count=u.get("prompt_tokens", 0),
        candidates_token_count=u.get("completion_tokens", 0),
        total_token_count=u.get("total_tokens", 0),
    )

    # Mostrar en pantalla
    print(f"Prompt:  {usage.prompt_token_count} tokens")
    print(f"Salida:  {usage.candidates_token_count} tokens")
    print(f"Total:   {usage.total_token_count} tokens")

    # Guardado CSV (sin cambios)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    col_name = f"N{N}_k{k}_{timestamp}"
    rows = ["tokens_prompt", "tokens_candidates", "tokens_thoughts", "tokens_total", "results"]
    values = [
        usage.prompt_token_count,
        usage.candidates_token_count,
        0,  # no hay 'thoughts' separado
        usage.total_token_count,
        "ok"
    ]

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        df = df.reindex(index=rows)
    else:
        df = pd.DataFrame(index=rows)

    df[col_name] = values
    df.to_csv(csv_path)

    return content


import json

def extract_solution_from_text(text: str) -> List[List[str]]:
    """
    Extrae la lista de movimientos de un texto, siguiendo esta lógica:
    - Mantiene solo lo que está entre el primer `[` y el último `]`.
    - Elimina caracteres innecesarios como saltos, comentarios, símbolos no válidos.
    - Evalúa la estructura como una lista de listas de strings.
    """
    # Paso 1: encontrar límites del bloque de lista
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("❌ No se encontraron corchetes válidos.")

    raw_block = text[start:end + 1]

    # Paso 2: eliminar caracteres no deseados
    # Permitimos solo: letras, números, _, comas, comillas, corchetes y espacios mínimos
    cleaned_block = re.sub(r"[^a-zA-Z0-9_\[\]\",']", "", raw_block)

    # Paso 3: evaluar
    try:
        result = eval(cleaned_block)
        if not isinstance(result, list) or not all(isinstance(m, list) for m in result):
            raise ValueError("❌ El formato no es una lista de listas.")
        return result
    except Exception as e:
        raise ValueError(f"❌ No se pudo evaluar la lista: {e}")


# ------------------ Uso (sin cambios) ------------------
N = 2  # Number of jealous couples
k = 2  # Capacity of the boat

# Paso 1: Construir el prompt
prompt = build_river_crossing_prompt(N=N, k=k)
print(f"Prompt generado:\n{prompt}\n")

# Paso 2: Llamar al "modelo" (ahora Ollama local, mismo nombre de función)
respuesta = call_gemini_model(prompt, N, k)
print(f"Respuesta del modelo:\n{respuesta}\n")

# Paso 3: Extraer la solución
moves = extract_solution_from_text(respuesta)
print(moves)
