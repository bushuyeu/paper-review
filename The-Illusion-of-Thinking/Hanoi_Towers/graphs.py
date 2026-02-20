import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# Cargar CSV corregido
df = pd.read_csv("results/hanoi_token_usage_conver.csv")

# Extraer configuración base (N y p) desde 'Name'
df['config'] = df['Name'].str.extract(r'(N\d+_p\d+)')
df['N'] = df['Name'].str.extract(r'N(\d+)_').astype(int)
df['p'] = df['Name'].str.extract(r'_p(\d+)').astype(int)

# Calcular tasa de éxito por configuración
success_rate = df.groupby('config')['results'].apply(lambda x: (x == 'ok').mean() * 100)

# Calcular suma de tokens totales y media/desviación
token_cols = [f'tokens_total_iter{i}' for i in range(1, 11)]
df['total_tokens'] = df[token_cols].sum(axis=1)
token_stats = df.groupby('config')['total_tokens'].agg(['mean', 'std'])

# Calcular media y desviación de tokens por iteración
df['tokens_per_iter'] = df['total_tokens'] / 10
tokens_per_iter_stats = df.groupby('config')['tokens_per_iter'].agg(['mean', 'std'])

# Ordenar configuraciones según N y p
config_order = df[['config', 'N', 'p']].drop_duplicates().sort_values(by=['N', 'p'])
configs_sorted = config_order['config'].values
x = np.arange(len(configs_sorted))

# Crear etiquetas del eje X como N=...; p=...
x_labels = [f"N={row.N}; p={row.p}" for row in config_order.itertuples()]

# Crear gráfico
fig, ax1 = plt.subplots(figsize=(16, 12))


# Eje 1: Success Rate
ax1.bar(x, success_rate[configs_sorted], color='#cdb4db', width=0.4, label='Success Rate (%)')
ax1.set_ylabel('Success Rate (%)', fontsize=24)
ax1.set_ylim(0, 110)
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=20)
ax1.tick_params(axis='y', labelsize=22, labelcolor='purple')
ax1.tick_params(axis='x', labelsize=22, labelcolor='black')


# Eje 2: Tokens
ax2 = ax1.twinx()

# Media total de tokens con desviación
ax2.errorbar(
    x, token_stats.loc[configs_sorted, 'mean'],
    yerr=token_stats.loc[configs_sorted, 'std'],
    fmt='o-', capsize=5, capthick=3, elinewidth=3, linewidth=5, markersize=13,
    color='#3d3d3d', label='Total Tokens ± STD'
)


# Media tokens por iteración con desviación
ax2.errorbar(
    x, tokens_per_iter_stats.loc[configs_sorted, 'mean'],
    yerr=tokens_per_iter_stats.loc[configs_sorted, 'std'],
    fmt='s--', capsize=5, capthick=2, elinewidth=2, linewidth=4,
    color="#80a3e4", label='Avg. Tokens per Request ± STD'
)

# Crear zoom (inset) más pequeño y con ejes visibles
axins = inset_axes(ax2, width="45%", height="15%", loc='upper right', borderpad=0.8)

# Curva dentro del inset
axins.errorbar(
    x, tokens_per_iter_stats.loc[configs_sorted, 'mean'],
    yerr=tokens_per_iter_stats.loc[configs_sorted, 'std'],
    fmt='s--', capsize=5, capthick=2.2, elinewidth=1.9, linewidth=3,
    color="#80a3e4"
)

# Limites del zoom
axins.set_ylim(-1000, 13000)
axins.set_xlim(x[0] - 0.5, x[-1] + 0.5)
axins.set_xticks(x)
axins.set_xticklabels(x_labels, rotation=50, ha='right', fontsize=10)
axins.set_yticks([0, 3000, 6000, 9000, 12000])
axins.tick_params(axis='both', labelsize=16)
# axins.set_title("Zoom: Avg. Tokens per Request", fontsize=8)

# Opcional: eliminar líneas de conexión si molestan
# mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="0.5")


# Marcar región del zoom (no relleno, borde gris claro)
mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="0.5")

ax2.set_ylabel('Tokens', fontsize=24)
ax2.tick_params(axis='y', labelsize=22, labelcolor='black')
ax2.set_ylim(0, 160000)


# Leyenda y estilo
fig.tight_layout(rect=[0, 0, 1, 0.95])

# Leyenda en la esquina superior izquierda, dentro del gráfico
fig.legend(
    loc='upper left',
    bbox_to_anchor=(0.01, 0.99),  # Cerca del borde pero dentro
    bbox_transform=ax1.transAxes,
    fontsize=19
)

plt.suptitle('Towers of Hanoi - Agentic Dialogue', fontsize=26, fontweight='bold', y=1.02)
plt.grid(False)


# Guardar la figura
os.makedirs("results", exist_ok=True)
# Guardar success rate en CSV
success_rate_df = success_rate.loc[configs_sorted].reset_index()
success_rate_df.columns = ['config', 'success_rate_percent']
success_rate_df.to_csv("results/conver_hanoi_success_rate.csv", index=False)

plt.savefig("results/hanoi_token_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

# Mostrar resumen estructurado
print("\nResumen de uso total de tokens por configuración:\n")
print(f"{'Configuración':<15} {'Media Total':>12} {'Desv. Total':>15} {'Media/Iter':>15} {'Desv/Iter':>15}")
print("-" * 70)
for config in configs_sorted:
    mean_total = token_stats.loc[config, 'mean']
    std_total = token_stats.loc[config, 'std']
    mean_iter = tokens_per_iter_stats.loc[config, 'mean']
    std_iter = tokens_per_iter_stats.loc[config, 'std']
    print(f"{config:<15} {mean_total:12.1f} {std_total:15.2f} {mean_iter:15.2f} {std_iter:15.2f}")
