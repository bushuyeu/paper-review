import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Cargar tu CSV
df = pd.read_csv("results/river_success_rate.csv")

# Extraer N y k
df['N'] = df['config'].str.extract(r'N=(\d+)').astype(int)
df['k'] = df['config'].str.extract(r'k=(\d+)').astype(int)
df['key'] = df['N'].astype(str) + 'k' + df['k'].astype(str)
df['label'] = df.apply(lambda row: f"N={row['N']}; k={row['k']}", axis=1)

# Resultados de Shojaee et al.
claude_dict = {'2k2': 82, '3k2': 0, '4k3': 0, '5k3': 0, '5k4': 0, '10k4': 0, '20k4': 0, '50k4': 0, '100k4': 0}
deepseek_dict = {'2k2': 92, '3k2': 0, '4k3': 0, '5k3': 0, '5k4': 0, '10k4': 0, '20k4': 0, '50k4': 0, '100k4': 0}

# Eje x
keys = df['key'].tolist()
x = np.arange(len(keys))
labels = df['label'].tolist()
ours = df['success_rate_percent'].values
claude = [claude_dict.get(k, np.nan) for k in keys]
deepseek = [deepseek_dict.get(k, np.nan) for k in keys]

# Colores
color_ours = "#bf38e1"
color_claude = "#8bcf88"     # Lavanda pastel
color_deepseek = "#8bd3e6"   # Melocotón pastel


# Crear gráfico
fig, ax = plt.subplots(figsize=(14, 7))

# Zonas de dificultad ajustadas con precisión
ax.axvspan(-0.5, 1.0, facecolor='#d0f0c0', alpha=0.3, zorder=0)   # Verde: hasta N=3; k=2 (índice 1 incluido)
ax.axvspan(1.0, 2.0, facecolor='#fff3b0', alpha=0.3, zorder=0)    # Amarillo: hasta N=4; k=3 (índice 2 incluido)
ax.axvspan(2.0, 4.0, facecolor='#f4cccc', alpha=0.3, zorder=0)    # Rojo: hasta N=5; k=4 (índice 4 incluido)
ax.axvspan(4.0, len(x)-0.5, facecolor='#d0f0c0', alpha=0.3, zorder=0)  # Verde desde ahí hasta el final

# Dibujar líneas
ax.plot(x, ours, 'o-', linewidth=3, color=color_ours, label="Ours", markersize=8, zorder=10)
ax.plot(x, claude, 'D--', linewidth=3, color=color_claude, label="Apple - Claude 3.7 Sonnet", markersize=8, zorder=9)
ax.plot(x, deepseek, 'X--', linewidth=3, color=color_deepseek, label="Apple - DeepSeek-R1", markersize=8, zorder=8)

# Ejes y etiquetas
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=16)
ax.set_xlim(-0.5, len(x) - 0.5)

ax.set_xlabel("Configuration (N = number of agents, k = boat capacity)", fontsize=22, fontweight='bold')
ax.set_ylabel("Success Rate (%)", fontsize=22, fontweight='bold')
ax.set_ylim(-5, 110)
ax.set_title("River Crossing – Comparison with Shojaee et al.", fontsize=24, fontweight='bold')
ax.tick_params(axis='y', labelsize=22)
ax.tick_params(axis='x', labelsize=26)
ax.legend(fontsize=24)
plt.grid(False)

# Guardar
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.2)
plt.savefig("results/river_success_comparison.png", dpi=300,bbox_inches='tight')
plt.show()
