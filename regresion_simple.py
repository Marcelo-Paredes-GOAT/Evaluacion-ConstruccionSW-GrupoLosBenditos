"""
PARTE 2.1 - Regresión Lineal Simple
Rama: feat/regresion-simple
Dataset A: Precios de viviendas
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ─── Cargar datos ───────────────────────────────────────────────────────────
df = pd.read_csv('../data/dataset_A_viviendas.csv')
print("=== Dataset A - Viviendas ===")
print(df.describe())

# ─── Correlación ────────────────────────────────────────────────────────────
correlaciones = df.corr()['precio_usd'].drop('precio_usd').abs()
mejor_var = correlaciones.idxmax()
print(f"\nVariable con mayor correlación con precio: '{mejor_var}' (r = {correlaciones[mejor_var]:.4f})")

# ─── Preparar datos ──────────────────────────────────────────────────────────
X = df[[mejor_var]].values
y = df['precio_usd'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─── Entrenar modelo ─────────────────────────────────────────────────────────
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# ─── Métricas ────────────────────────────────────────────────────────────────
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2  = r2_score(y_test, y_pred)

print(f"\n=== Regresión Lineal Simple ===")
print(f"Variable independiente : {mejor_var}")
print(f"Coeficiente (pendiente): {modelo.coef_[0]:.4f}")
print(f"Intercepto             : {modelo.intercept_:.4f}")
print(f"MSE                    : {mse:,.2f}")
print(f"RMSE                   : {rmse:,.2f}")
print(f"R²                     : {r2:.4f}")

# ─── Gráfica ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Regresión Lineal Simple — Precio de Viviendas', fontsize=14, fontweight='bold')

# 1. Dispersión + línea de tendencia
x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_range = modelo.predict(x_range)

axes[0].scatter(X_test, y_test, alpha=0.5, color='steelblue', label='Datos reales', s=30)
axes[0].plot(x_range, y_range, color='red', linewidth=2, label=f'Regresión (R²={r2:.3f})')
axes[0].set_xlabel(mejor_var.replace('_', ' ').title())
axes[0].set_ylabel('Precio (USD)')
axes[0].set_title('Línea de Tendencia')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Real vs Predicho
axes[1].scatter(y_test, y_pred, alpha=0.5, color='darkorange', s=30)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
axes[1].plot(lims, lims, 'k--', linewidth=1.5, label='Predicción perfecta')
axes[1].set_xlabel('Precio Real (USD)')
axes[1].set_ylabel('Precio Predicho (USD)')
axes[1].set_title('Real vs Predicho')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../graficas/regresion_simple.png', dpi=120, bbox_inches='tight')
plt.close()
print("\nGráfica guardada: graficas/regresion_simple.png")

# Guardar métricas para el informe
metricas = {'modelo': 'Regresion Lineal Simple', 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'variable': mejor_var}
print("\nMétricas exportadas:", metricas)
