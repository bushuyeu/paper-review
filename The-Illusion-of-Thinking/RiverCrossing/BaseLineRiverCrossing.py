import os
import re
import ast
from typing import List
import google.generativeai as genai
from RiverCrossingViewer import RiverCrossingVisualizer
import pandas as pd
from datetime import datetime

# CONFIGURA TU API KEY
genai.configure(api_key=os.getenv("GEMINI_API_KEY_HANOI"))


def build_river_crossing_prompt(N: int, k: int) -> str:
    """
    Build the river-crossing statement WITHOUT checking solvability.
    This is intentionally done to match the original paper conditions
    where unsolvable instances were used.
    """
    if k < 1:
        raise ValueError("❌ Boat capacity k must be at least 1.")
    if N < 1:
        raise ValueError("❌ There must be at least one actor and one agent (N ≥ 1).")

    # NO solvability checks - this is intentional for baseline comparison
    return (
        f"{N} actors and their {N} agents want to cross a river in a boat that is capable of holding "
        f"only {k} people at a time, with the constraint that no actor can be in the presence of another agent, "
        f"including while riding the boat, unless their own agent is also present, because each agent is worried "
        f"their rivals will poach their client. Initially, all actors and agents are on the left side of the river "
        f"with the boat. How should they cross the river? (Note: the boat cannot travel empty)"
    )


def call_gemini_model(
    prompt_text: str,
    N: int,
    k: int,
    csv_path: str = "tokens_river_baseline.csv",
    model_name: str = "gemini-2.5-pro-preview-06-05"
) -> tuple[str, object]:
    """
    Call Gemini model with the exact system and user prompts specified.
    Returns both the response text and usage metadata.
    """
    # System prompt - exact copy as requested
    system_instruction = (
        "You are a helpful assistant. Solve this puzzle for me. You can represent actors with a_1, a_2, ... "
        "and agents with A_1, A_2, ... . Your solution must be a list of boat moves where each move indicates "
        "the people on the boat. For example, if there were two actors and two agents, you should return: "
        "moves=[[\"A_2\", \"a_2\"], [\"A_2\"], [\"A_1\", \"A_2\"], [\"A_1\"], [\"A_1\", \"a_1 \"]] which indicates "
        "that in the first move, A_2 and a_2 row from left to right, and in the second move, A_2 rows from right "
        "to left and so on. Requirements: • The list shouldn't have comments. • Ensure your "
        "final answer also includes the complete list of moves for final solution. "
        "IMPORTANT: Your response must have the correct format as this is vital for the evaluation."
    )
    
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction
    )
    
    response = model.generate_content(prompt_text)
    usage = response.usage_metadata

    # Mostrar en pantalla
    print(f"Prompt:  {usage.prompt_token_count} tokens")
    print(f"Salida:  {usage.candidates_token_count} tokens")
    print(f"Total:   {usage.total_token_count} tokens")

    # Preparar datos CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    col_name = f"N{N}_k{k}_{timestamp}"
    rows = ["tokens_prompt", "tokens_candidates", "tokens_thoughts", "tokens_total", "results"]
    values = [
        usage.prompt_token_count,
        usage.candidates_token_count,
        getattr(usage, "thoughts_token_count", 0),
        usage.total_token_count,
        "baseline"  # Mark as baseline experiment
    ]

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        # Ensure all rows exist
        df = df.reindex(index=rows)
    else:
        df = pd.DataFrame(index=rows)

    df[col_name] = values
    df.to_csv(csv_path)

    return response.text, usage


def extract_solution_from_text(text: str) -> List[List[str]]:
    """
    Extrae la lista de movimientos de un texto usando múltiples estrategias robustas.
    Busca patrones como 'moves=[...]' o simplemente listas aisladas.
    """
    
    # Estrategia 1: Buscar 'moves = [...]' explícitamente
    moves_pattern = re.search(r'moves\s*=\s*(\[.*?\])', text, re.DOTALL | re.IGNORECASE)
    if moves_pattern:
        try:
            moves_text = moves_pattern.group(1)
            # Limpiar comentarios y caracteres extraños
            cleaned_text = re.sub(r'#.*', '', moves_text)  # Remover comentarios
            cleaned_text = re.sub(r'[^\w\[\],"\'_\s]', '', cleaned_text)  # Solo chars válidos
            
            result = ast.literal_eval(cleaned_text)
            if isinstance(result, list) and all(isinstance(m, list) for m in result):
                return result
        except Exception:
            pass  # Continuar con la siguiente estrategia
    
    # Estrategia 2: Buscar la última lista completa en el texto
    # Encuentra todos los bloques que empiecen con [ y terminen con ]
    list_blocks = re.findall(r'\[(?:[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*)*\]', text)
    
    for block in reversed(list_blocks):  # Empezar por el último (más probable)
        try:
            # Limpiar el bloque
            cleaned_block = re.sub(r'[^a-zA-Z0-9_\[\]\",\'\s]', '', block)
            result = ast.literal_eval(cleaned_block)
            
            # Verificar que sea una lista de listas válida para movimientos
            if (isinstance(result, list) and 
                len(result) > 0 and 
                all(isinstance(m, list) for m in result) and
                all(len(m) > 0 for m in result) and
                all(isinstance(item, str) for m in result for item in m)):
                return result
        except Exception:
            continue
    
    # Estrategia 3: Buscar entre el primer [ y último ] (método original mejorado)
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        try:
            raw_block = text[start:end + 1]
            # Limpieza más agresiva
            cleaned_block = re.sub(r'[^a-zA-Z0-9_\[\]\",\'\s]', '', raw_block)
            
            result = ast.literal_eval(cleaned_block)
            if isinstance(result, list) and all(isinstance(m, list) for m in result):
                return result
        except Exception:
            pass
    
    # Si nada funciona, lanzar error
    raise ValueError("❌ No se pudo extraer una lista válida de movimientos del texto")


def save_results_to_csv(N: int, k: int, success: bool, usage_metadata=None, 
                       csv_path: str = "results/river_crossing_baseline.csv"):
    """
    Guarda los resultados del experimento en un archivo CSV en la carpeta results,
    siguiendo el mismo formato que los otros puzzles.
    """
    # Crear directorio results si no existe
    os.makedirs("results", exist_ok=True)
    
    # Timestamp para el experimento
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Nombre único para esta ejecución
    name = f"N{N}_k{k}_{timestamp}"
    
    # Resultado (ok/fail)
    result = "ok" if success else "fail"
    
    # Preparar fila de datos
    new_row = {
        "Name": name,
        "N": N,
        "k": k,
        "tokens_prompt": usage_metadata.prompt_token_count if usage_metadata else 0,
        "tokens_candidates": usage_metadata.candidates_token_count if usage_metadata else 0,
        "tokens_total": usage_metadata.total_token_count if usage_metadata else 0,
        "results": result
    }
    
    # Cargar CSV existente o crear uno nuevo
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Agregar la nueva fila
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        # Crear nuevo DataFrame
        df = pd.DataFrame([new_row])
    
    # Guardar el CSV
    df.to_csv(csv_path, index=False)
    
    print(f"📁 Resultados guardados en: {csv_path}")
    return csv_path


if __name__ == "__main__":
    # Configuración del experimento baseline
    N = 8  # Number of jealous couples
    k = 3  # Capacity of the boat
    
    print("🧪 BASELINE RIVER CROSSING EXPERIMENT")
    print("=" * 50)
    print(f"⚠️  WARNING: Using potentially unsolvable configuration (N={N}, k={k})")
    print("   This matches the original paper conditions.")
    print("=" * 50)
    
    # Paso 1: Construir el prompt para N actores/agentes y k capacidad del bote
    # (SIN verificación de solvabilidad)
    prompt = build_river_crossing_prompt(N=N, k=k)
    print(f"📝 Prompt generado:\n{prompt}\n")

    # Paso 2: Llamar al modelo Gemini con ese prompt
    print("🤖 Llamando al modelo Gemini...")
    respuesta, usage_metadata = call_gemini_model(prompt, N, k)
    print(f"📋 Respuesta del modelo:\n{respuesta}\n")

    # Variables para guardar resultados
    success = False
    moves = None
    error_message = None

    # Paso 3: Intentar extraer la solución (lista de movimientos) del texto generado
    try:
        moves = extract_solution_from_text(respuesta)
        print("✅ Movimientos extraídos:")
        print(moves)
        print()
        
        # Paso 4: Validar movimientos sin generar video
        print("🔍 Validando movimientos...")
        try:
            viz = RiverCrossingVisualizer(N, k, moves)
            # Solo validar sin animar
            is_valid = viz.is_valid_solution()
            if is_valid:
                print("✅ Los movimientos son válidos y resuelven el puzzle.")
                success = True
            else:
                print("❌ Los movimientos contienen errores o no resuelven el puzzle.")
                error_message = "Los movimientos no resuelven el puzzle correctamente"
        except Exception as validation_error:
            print(f"❌ Error en validación: {validation_error}")
            error_message = f"Error en validación: {validation_error}"
        
    except ValueError as e:
        print(f"❌ Error al extraer solución: {e}")
        error_message = f"Error al extraer solución: {e}"
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        error_message = f"Error inesperado: {e}"
    
    # Guardar resultados en CSV
    save_results_to_csv(N, k, success, usage_metadata)
    
    print("\n" + "=" * 50)
    print("📊 Experimento baseline completado.")
    print("   Los resultados se han guardado en tokens_river_baseline.csv")
    print("   y en results/river_crossing_baseline.csv")
