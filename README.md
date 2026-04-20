
# Evaluación: Construcción de Software - Grupo Los Benditos

## Integrantes del Equipo
JOEL MILTON QUISPE CRUZ
MARCELO FLAVIO PAREDES ESTRADA
ANTONI ALDAIR LERMA CONDORI
CALLANAUPA CJUIRO MANUEL FABRIZIO
JANCCO SALAS JUAN SEBASTIAN

## Instrucciones de Ejecución
Para ejecutar los modelos desarrollados en este proyecto, asegúrese de tener instaladas las bibliotecas `scikit-learn`, `pandas`, `numpy` y `matplotlib`.

1. **Preparación**: Asegúrese de que los archivos `train.csv` (Dataset A) y el archivo de pacientes (Dataset B) estén en la misma carpeta que los scripts.
2. **Ejecución de Regresiones**:
   - Ejecute `Regresion_Lineal_Simple.py` para ver la tendencia base.
   - Ejecute `Regresion_Lineal_Multiple.py` para el análisis con múltiples variables.
   - Ejecute el script de Regresión Polinómica para observar ajustes no lineales.
3. **Ejecución de Clasificación**:
   - Ejecute el script de Regresión Logística para obtener la matriz de confusión y métricas de salud.

## Conclusiones
Tras el análisis de los modelos de regresión, se concluye que la **Regresión Lineal Múltiple** tuvo el mejor rendimiento para predecir los precios de las viviendas. [cite_start]Esto se evidencia en su coeficiente $R^2$ de **0.706**, el más alto de los tres modelos, y su MSE más bajo, lo que indica que incluir múltiples variables independientes permite capturar mejor la variabilidad de los datos que un modelo simple o polinómico de una sola variable[cite: 29, 50].

En cuanto al modelo de clasificación, la **Regresión Logística** demostró ser altamente efectiva para el historial médico, alcanzando una exactitud (Accuracy) del **95%**. [cite_start]La matriz de confusión y el alto nivel de Precision/Recall confirman que el preprocesamiento de datos y el escalado de variables permitieron una predicción robusta para identificar correctamente a los pacientes[cite: 42, 50].
