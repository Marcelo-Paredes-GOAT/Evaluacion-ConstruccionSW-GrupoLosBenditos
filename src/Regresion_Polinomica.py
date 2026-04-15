# =====================================================================
# REGRESIÓN POLINÓMICA
# =====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

print("--- 1. ENTRENAMIENTO CON DATOS REALES (train.csv) ---")
# Cargar datos
df = pd.read_csv('train.csv')
df = df.select_dtypes(include=[np.number]).dropna()

# Elegimos GrLivArea (Área Habitable) porque el precio de las casas inmensas 
# tiende a dispararse exponencialmente (ideal para un polinomio)
X_real = df[['GrLivArea']]
y_real = df['SalePrice']

# Dividimos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42)

# Transformamos nuestros datos reales a Polinómicos (Grado 2: añade la columna x^2)
grado = 2
poly = PolynomialFeatures(degree=grado, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Entrenamos el modelo lineal usando estas nuevas características al cuadrado
modelo_poly = LinearRegression()
modelo_poly.fit(X_train_poly, y_train)

# Calculamos las métricas reales
y_pred_real = modelo_poly.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred_real)
r2 = r2_score(y_test, y_pred_real)

# Extraemos los coeficientes de la ecuación real
c = modelo_poly.intercept_
b = modelo_poly.coef_[0]  # Peso de X (Área normal)
a = modelo_poly.coef_[1]  # Peso de X^2 (Área al cuadrado)

print("Métricas del modelo polinómico sobre datos reales:")
print(f" - Error Cuadrático Medio (MSE): {mse:,.2f}")
print(f" - Coeficiente de Determinación (R^2): {r2:.4f}")
formula = f"Precio = {a:,.2f}x² + {b:,.2f}x + {c:,.0f}"
print(f" - Ecuación:\n   {formula}")


print("\n--- 2. GENERACIÓN DE CURVA SINTÉTICA PARA GRÁFICA ---")
# Generamos 100 puntos perfectamente ordenados desde la casa más pequeña a la más grande
x_min = X_real.min().values[0]
x_max = X_real.max().values[0]
x_sintetico = np.linspace(x_min, x_max, 100).reshape(-1, 1)

# Pasamos estos puntos sintéticos ordenados por el transformador polinómico y el modelo
x_sintetico_poly = poly.transform(x_sintetico)
y_sintetico = modelo_poly.predict(x_sintetico_poly)


print("\n--- 3. VISUALIZACIÓN ESTÉTICA ---")
plt.figure(figsize=(10, 6))

# 1. Dibujamos los puntos reales de prueba (La nube azul)
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Casas Reales (Test)')

# 2. Dibujamos la curva sintética perfecta (La línea roja suave)
# Usamos flatten() para asegurarnos de que matplotlib no tenga problemas de formato
plt.plot(x_sintetico.flatten(), y_sintetico, color='red', linewidth=3, label=f'Curva Polinómica (Grado {grado})')

# Personalización
plt.title('Regresión Polinómica: Área Habitable vs Precio')
plt.xlabel('Área Habitable (GrLivArea)')
plt.ylabel('Precio de Venta Estimado ($)')

# Imprimimos la fórmula en pantalla
plt.text(0.05, 0.90, formula, transform=plt.gca().transAxes, fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.legend()
plt.grid(True)
plt.show()

print("\nAnálisis (Para el informe): Al aplicar el polinomio de grado 2, la curva roja logra capturar la aceleración del precio en las casas de mayor tamaño, ajustándose mejor a la realidad que una simple línea recta.")