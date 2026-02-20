import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Asegurar carpeta
os.makedirs("results", exist_ok=True)

# Cargar CSVs
steps_df = pd.read_csv("results/steps_success_rate.csv")
conver_df = pd.read_csv("results/conver_hanoi_success_rate.csv")

# Extraer N
steps_df["N"] = steps_df["config"].str.extract(r'N(\d+)').astype(int)
conver_df["N"] = conver_df["config"].str.extract(r'N(\d+)').astype(int)

# Datos Apple
apple_claude = {3: 97, 4: 94, 5: 91, 6: 88, 7: 70, 8: 10, 10: 0}
apple_deepseek = {3: 93, 4: 95, 5: 91, 7: 72, 8: 10, 10: 3}

# Nuevo rango X desde N=3 hasta 10
x = np.arange(3, 11)

# Construir listas
def fill_series(df, column):
    return [df[df["N"] == n][column].values[0] if n in df["N"].values else np.nan for n in x]

steps_rate = fill_series(steps_df, "success_rate_percent")
conver_rate = fill_series(conver_df, "success_rate_percent")
claude_rate = [apple_claude.get(n, np.nan) for n in x]
deepseek_rate = [apple_deepseek.get(n, np.nan) for n in x]

# Interpolación para conectar líneas sin extender más allá
def interpolate(series):
    return pd.Series(series, index=x).interpolate(limit_direction='both').tolist()

steps_rate = interpolate(steps_rate)
conver_rate = interpolate(conver_rate)
claude_rate = interpolate(claude_rate)
deepseek_rate = interpolate(deepseek_rate)

# Plot
fig, ax = plt.subplots(figsize=(14, 9))

# Zonas de dificultad ajustadas
ax.axvspan(3, 4, facecolor='#d0f0c0', alpha=0.4)   # Verde
ax.axvspan(4, 8, facecolor='#fff3b0', alpha=0.4)   # Amarillo
ax.axvspan(8, 10.5, facecolor='#f4cccc', alpha=0.4)  # Rojo

# Gráficas
ax.plot(x, steps_rate, 'o-', linewidth=3, color='#4f9ddb', label='Ours - Stepwise', markersize=8)
ax.plot(x, conver_rate, 's-', linewidth=3, color='#f97c7c', label='Ours - Agentic Dialogue', markersize=8)
ax.plot(x, claude_rate, 'D--', linewidth=3, color='#8bcf88', label='Apple - Claude 3.7 Sonnet', markersize=8)
ax.plot(x, deepseek_rate, 'X--', linewidth=3, color='#8bd3e6', label='Apple - DeepSeek R1', markersize=8)

# Estética
ax.set_xlabel("Number of Disks (N)", fontsize=22, fontweight='bold')
ax.set_ylabel("Success Rate (%)", fontsize=22, fontweight='bold')
ax.set_title("Towers of Hanoi - Comparison with Shojaee et al.", fontsize=26, fontweight='bold')
ax.set_xticks(x)
ax.set_xlim(3, 10)
ax.set_ylim(0, 110)
ax.tick_params(axis='both', which='major', labelsize=22)
ax.legend(fontsize=24, loc='upper right')
ax.grid(False)

# Guardar y mostrar
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.2)
plt.savefig("results/hanoi_success_comparison.png", dpi=300,bbox_inches='tight')
plt.show()
