import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Ruta al archivo CSV
file_path = "tokens_river.csv"  # Asegúrate de que el archivo está en el mismo directorio o pon la ruta completa

# Cargar el CSV y reestructurar
df_raw = pd.read_csv(file_path)
df = df_raw.set_index(df_raw.columns[0])
df = df.drop(columns=df.columns[0])

# Extraer las configuraciones experimentales
columns = df.columns
experiments = {}

for col in columns:
    match = re.match(r'N(\d+)_k(\d+)_\d+', col)
    if match:
        key = f'N={match.group(1)} k={match.group(2)}'
        if key not in experiments:
            experiments[key] = []
        experiments[key].append(col)

# Inicializar estructuras
means = []
stds = []
labels = []
ok_percentages = []
fail_percentages = []

# Calcular estadísticas
for key, cols in experiments.items():
    tokens_total = df.loc["tokens_total", cols].astype(float)
    results = df.loc["results", cols]

    mean = tokens_total.mean()
    std = tokens_total.std()
    ok_percentage = (results == "ok").sum() / len(results)
    fail_percentage = 1 - ok_percentage

    labels.append(key)
    means.append(mean)
    stds.append(std)
    ok_percentages.append(ok_percentage)
    fail_percentages.append(fail_percentage)

# Colores pastel
pastel_green = "#b6e6bd"
pastel_red = "#f7c6c7"
dark_gray = "#3c3c3c"

# Crear figura
fig, ax1 = plt.subplots(figsize=(14, 7))
x = np.arange(len(labels))
bar_width = 0.5

# Dibujar fondo de éxito/fracaso (debajo de la línea)
max_token = max([m + s for m, s in zip(means, stds)]) * 1.1
for i, (ok, fail) in enumerate(zip(ok_percentages, fail_percentages)):
    ax1.bar(i, ok * max_token, width=bar_width, color=pastel_green, zorder=0)
    ax1.bar(i, fail * max_token, bottom=ok * max_token,
            width=bar_width, color=pastel_red, zorder=0)

# Dibujar línea con barras de error
ax1.errorbar(x, means, yerr=stds, fmt='-o', color=dark_gray, capsize=6,
             linewidth=3, markersize=10, zorder=5)

# Etiquetas y estilos
ax1.set_xlabel("Experiment Configuration (N = number of couples, k = boat capacity)", fontsize=14)
ax1.set_ylabel("Mean Tokens", fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
ax1.tick_params(axis='y', labelsize=12)

# Eje derecho con porcentaje de acierto
ax2 = ax1.twinx()
ax2.set_ylim(0, 1)
ax2.set_ylabel("Success Rate", fontsize=14)
ax2.set_yticks(np.linspace(0, 1, 5))
ax2.set_yticklabels([f"{int(y*100)}%" for y in np.linspace(0, 1, 5)], fontsize=12)
ax2.tick_params(axis='y', labelsize=12)

plt.title("Mean Tokens per Experiment Configuration\nwith Success/Failure Proportion",
          fontsize=16, fontweight='bold')

plt.grid(False)
# Crear leyenda personalizada
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color=dark_gray, lw=3, marker='o', markersize=10, label='Mean Tokens ± STD'),
    Patch(facecolor=pastel_green, edgecolor='none', label='Success Rate'),
    Patch(facecolor=pastel_red, edgecolor='none', label='Failure Rate')
]

ax1.legend(handles=legend_elements, loc='upper left', fontsize=12)

import os
os.makedirs("results", exist_ok=True)

# Guardar success rates en CSV
success_df = pd.DataFrame({
    "config": labels,
    "success_rate_percent": [round(ok * 100, 2) for ok in ok_percentages]
})

success_df.to_csv("results/river_success_rate.csv", index=False)


plt.tight_layout()
plt.savefig("river_results.jpeg", dpi=300)
plt.show()
