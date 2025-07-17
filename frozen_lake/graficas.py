import os
import pandas as pd
import matplotlib.pyplot as plt

# ====== CONFIGURACIÓN DEL USUARIO ======
carpeta_csv = "./csvs"  # Carpeta donde está el CSV
nombre_archivo_csv = "MeanReward.csv"  # Nombre del archivo CSV
nombre_eje_y = "Reward"
titulo_grafica = "Recompensa media a lo largo del entrenamiento"
factor_suavizado = 0.5  # Entre 0 y 1. Más bajo = más suave (como TensorBoard)

# Carpeta de salida
carpeta_graficas = os.path.join(carpeta_csv, "graficas")
os.makedirs(carpeta_graficas, exist_ok=True)

# Ruta completa al CSV
ruta_csv = os.path.join(carpeta_csv, nombre_archivo_csv)

# Verifica que exista
if not os.path.isfile(ruta_csv):
    print(f"[!] No se encontró el archivo: {ruta_csv}")
    exit()

# Leer CSV
df = pd.read_csv(ruta_csv)

if "Step" not in df.columns or "Value" not in df.columns:
    print(f"[!] El CSV debe contener las columnas 'Step' y 'Value'")
    exit()

# Obtener datos suavizados
x = df["Step"]
y = df["Value"].ewm(alpha=factor_suavizado).mean()

# Crear gráfica sin cuadrícula
plt.figure(figsize=(6, 4))
plt.plot(x, y, color='tab:blue', linewidth=1.5)
plt.xlabel("Number of Timesteps")
plt.ylabel(nombre_eje_y)
plt.title(titulo_grafica)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

# Sin grid
plt.grid(False)

# Guardar
nombre_salida = os.path.splitext(nombre_archivo_csv)[0] + f"_ppo_clean.png"
ruta_salida = os.path.join(carpeta_graficas, nombre_salida)
plt.savefig(ruta_salida, dpi=150)
plt.close()

print(f"[✓] Gráfica limpia guardada en: {ruta_salida}")