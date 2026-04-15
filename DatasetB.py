import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv('C:/Users/Estudiante/Downloads/PacientesMedicosDiabetes.csv')

# Separar características (X) y variable objetivo (y)
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Escalada de Características (Feature Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamiento del Modelo de Regresión Logística
model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced', max_iter=1000)
model.fit(X_train_scaled, y_train)
print('Modelo de Regresión Logística entrenado exitosamente.')

# Evaluación del Modelo
y_pred = model.predict(X_test_scaled)

# Calcular métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)



# Visualizar métricas de evaluación en tabla
metrics = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Valor': [accuracy, precision, recall, f1]
})
print(metrics)

# Matriz de Confusión
print('\n--- Matriz de Confusión ---')
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=['Real Negativo (0)', 'Real Positivo (1)'], columns=['Pred. Negativo (0)', 'Pred. Positivo (1)']))

# Visualizar la Matriz de Confusión
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabético (Pred.)', 'Diabético (Pred.)'],
            yticklabels=['No Diabético (Real)', 'Diabético (Real)'])
plt.title('Matriz de Confusión')
plt.ylabel('Valores Reales')
plt.xlabel('Valores Predichos')
plt.show()

print('\n--- Informe de Clasificación ---')
print(classification_report(y_test, y_pred))