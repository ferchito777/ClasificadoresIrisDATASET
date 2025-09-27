# ¿Cómo funciona DecisionTreeClassifier?
# DecisionTreeClassifier construye un árbol de decisión a partir de los datos de entrenamiento.
# El árbol realiza divisiones binarias en los datos usando las características más relevantes en cada nodo.
# Cada nodo interno representa una pregunta sobre una característica y cada hoja representa una clase predicha.
# El modelo busca maximizar la pureza de las hojas (que contengan ejemplos de una sola clase).
# Es fácil de interpretar visualmente y no requiere que los datos estén escalados.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap

# Cargar el dataset Iris
iris = load_iris()
X = iris.data[:, :2]  # Usamos solo las 2 primeras características para visualización
y = iris.target

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Crear y entrenar el clasificador Decision Tree
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Realizar predicciones
y_pred = clf.predict(X_test)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Mostrar métricas en consola
print("="*50)
print("MÉTRICAS DEL CLASIFICADOR")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("="*50)

# Crear la figura con 3 subplots en una fila
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Clasificador Decision Tree - Dataset Iris', fontsize=16, fontweight='bold')

# 1. Gráfica sin entrenar (datos originales)
ax1 = axes[0]
colors = ['red', 'blue', 'green']
target_names = iris.target_names

for i, color in enumerate(colors):
    ax1.scatter(X[y == i, 0], X[y == i, 1],
               c=color, marker='o',
               label=target_names[i],
               alpha=0.7, s=50)

ax1.set_xlabel('Longitud del sépalo (cm)')
ax1.set_ylabel('Anchura del sépalo (cm)')
ax1.set_title('Datos Originales\n(Sin entrenar)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Gráfica con fronteras de decisión entrenadas
ax2 = axes[1]

# Crear mesh para visualizar las fronteras de decisión
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predecir para cada punto del mesh
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Crear mapa de colores personalizado
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

# Plotear las fronteras de decisión
ax2.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

# Plotear los puntos de prueba
for i, color in enumerate(colors):
    ax2.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1],
               c=color, marker='o',
               label=target_names[i],
               alpha=0.8, s=60, edgecolor='black')

ax2.set_xlabel('Longitud del sépalo (cm)')
ax2.set_ylabel('Anchura del sépalo (cm)')
ax2.set_title('Fronteras de Decisión\n(Modelo Entrenado)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Matriz de confusión
ax3 = axes[2]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(ax=ax3, cmap='Blues', values_format='d')
ax3.set_title('Matriz de Confusión')

# Ajustar el layout
plt.tight_layout()
plt.show()