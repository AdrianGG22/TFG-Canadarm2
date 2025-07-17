import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ========== CONFIGURACIÓN ==========
carpeta_csv = "./csvs"
nombre_archivo_csv = "MeanRewardPendulum.csv"

titulo_grafica = "Recompensa media por episodio"
nombre_eje_x = "Number of Timesteps"
nombre_eje_y = "Rewards"

factor_suavizado = 0.1                     # 0.0 para sin suavizado
usar_notacion_cientifica = True            # Cambia a False para desactivar notación tipo 1e6
limites_sci = (3, 3)                       # (min, max) para activar notación tipo 1e6 justo en ese rango

figsize = (6, 4)
fontsize = 10
linewidth = 1.5

# ========== CARGA DE DATOS ==========
ruta_csv = os.path.join(carpeta_csv, nombre_archivo_csv)
if not os.path.isfile(ruta_csv):
    print(f"[!] No se encontró el archivo: {ruta_csv}")
    exit()

df = pd.read_csv(ruta_csv)

if "Step" not in df.columns or "Value" not in df.columns:
    print(f"[!] El CSV debe contener las columnas 'Step' y 'Value'")
    exit()

x = df["Step"]
y = df["Value"].ewm(alpha=factor_suavizado).mean() if factor_suavizado > 0 else df["Value"]

# ========== CREACIÓN DE GRÁFICA ==========
plt.figure(figsize=figsize)
plt.plot(x, y, color='tab:blue', linewidth=linewidth)
plt.xlabel(nombre_eje_x, fontsize=fontsize)
plt.ylabel(nombre_eje_y, fontsize=fontsize)
plt.title(titulo_grafica, fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

if usar_notacion_cientifica:
    formatter = ScalarFormatter(useMathText=False)  # No uses MathText para evitar ×10³
    formatter.set_powerlimits((3, 3))               # Fuerza uso a partir de mil
    formatter.set_useOffset(False)                  # Evita offsets raros
    plt.gca().xaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.grid(False)

# ========== GUARDAR ==========
carpeta_graficas = os.path.join(carpeta_csv, "graficas")
os.makedirs(carpeta_graficas, exist_ok=True)

nombre_salida = os.path.splitext(nombre_archivo_csv)[0]
ruta_salida = os.path.join(carpeta_graficas, nombre_salida + ".png")

plt.savefig(ruta_salida, dpi=150)
plt.close()

print(f"[✓] Gráfica guardada en: {ruta_salida}")
