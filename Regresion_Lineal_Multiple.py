# =====================================================================
# REGRESIÓN LINEAL MÚLTIPLE
# =====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("--- 1. ENTRENAMIENTO CON DATOS REALES (train.csv) ---")
# Cargar datos
df = pd.read_csv('train.csv')
df = df.select_dtypes(include=[np.number]).dropna()

# Elegimos 3 variables independientes (Requisito de la rúbrica)
X_real = df[['OverallQual', 'GrLivArea', 'GarageArea']]
y_real = df['SalePrice']

# Dividimos y entrenamos
X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
modelo_multi = LinearRegression()
modelo_multi.fit(X_train, y_train)

# Calculamos métricas reales
y_pred_real = modelo_multi.predict(X_test)
mse = mean_squared_error(y_test, y_pred_real)
r2 = r2_score(y_test, y_pred_real)

# Extraemos la ecuación real
b = modelo_multi.intercept_
w1, w2, w3 = modelo_multi.coef_

print("Métricas del modelo sobre datos reales:")
print(f" - Error Cuadrático Medio (MSE): {mse:,.2f}")
print(f" - Coeficiente de Determinación (R^2): {r2:.4f}")
formula = f"Precio = {b:,.0f} + ({w1:,.0f}*Qual) + ({w2:,.0f}*Area) + ({w3:,.0f}*Garage)"
print(f" - Ecuación:\n   {formula}")


print("\n--- 2. GENERACIÓN DE PLANO SINTÉTICO PARA GRÁFICA 3D ---")
# Para dibujar un plano suave, generamos 50x50 puntos sintéticos (2500 combinaciones)
# espaciados uniformemente desde el mínimo al máximo de nuestras variables reales.
x1_min, x1_max = X_real['OverallQual'].min(), X_real['OverallQual'].max()
x2_min, x2_max = X_real['GrLivArea'].min(), X_real['GrLivArea'].max()

x1_sintetico = np.linspace(x1_min, x1_max, 50)
x2_sintetico = np.linspace(x2_min, x2_max, 50)

# Creamos la cuadrícula (malla)
X1_malla, X2_malla = np.meshgrid(x1_sintetico, x2_sintetico)

# Para predecir en 3D, mantenemos la 3ra variable constante en su valor promedio real
x3_promedio = X_train['GarageArea'].mean()

# Calculamos el Precio (Z) sintético aplicando la ECUACIÓN REAL a nuestra malla
Z_malla = b + (w1 * X1_malla) + (w2 * X2_malla) + (w3 * x3_promedio)


print("\n--- 3. VISUALIZACIÓN 3D ESTÉTICA ---")
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 1. Dibujamos los puntos reales de prueba (La nube azul)
ax.scatter(X_test['OverallQual'], X_test['GrLivArea'], y_test, color='blue', alpha=0.5, label='Casas Reales (Test)')

# 2. Dibujamos el plano sintético continuo (El plano rojo suave)
ax.plot_surface(X1_malla, X2_malla, Z_malla, color='red', alpha=0.4)

# Personalización
ax.set_title('Regresión Múltiple: Predicción Continua en 3D')
ax.set_xlabel('Calidad General (OverallQual)')
ax.set_ylabel('Área Habitable (GrLivArea)')
ax.set_zlabel('Precio Estimado ($)')

# Imprimimos la fórmula en pantalla
ax.text2D(0.05, 0.95, formula, transform=ax.transAxes, fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# Giramos un poco la cámara para que el plano se vea espectacular
ax.view_init(elev=20, azim=135)
plt.legend()
plt.show()