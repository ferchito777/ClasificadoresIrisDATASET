"""
# ¿Cómo funciona KNeighborsClassifier?
# KNeighborsClassifier (KNN) es un algoritmo de clasificación basado en la proximidad.
# Para predecir la clase de una muestra nueva, busca los 'k' vecinos más cercanos en el conjunto de entrenamiento
# (usando una métrica de distancia, normalmente Euclidiana) y asigna la clase más común entre esos vecinos.
# No requiere entrenamiento explícito, solo almacena los datos y compara distancias.
# Es sencillo, interpretable y funciona bien cuando las clases están bien separadas en el espacio de características.
# """
# Importar librerí­as necesarias
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Crear una instancia del clasificador KNN
knn = KNeighborsClassifier()

# Cargar el dataset iris
iris = load_iris()

# Usar solo las dos primeras características para visualización y entrenamiento
X = iris.data[:, :2]
y = iris.target  # Etiquetas (clases)


# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Entrenar el modelo con los datos de entrenamiento
knn.fit(X_train, y_train)

# Realizar predicciones sobre el conjunto de entrenamiento
y_pred_train = knn.predict(X_train)

# Realizar predicciones sobre el conjunto de prueba
y_pred_test = knn.predict(X_test)

# Calcular la precisión positiva (precision) del modelo en el conjunto de prueba
precision = precision_score(y_test, y_pred_test, average='macro')
print(f'Precision en prueba: {precision:.2f}')

# Calcular la precisión global (accuracy) del modelo en el conjunto de prueba
accuracy = accuracy_score(y_test, y_pred_test)
print(f'Accuracy en prueba: {accuracy:.2f}')

# Calcular el F1 score del modelo en el conjunto de prueba
f1 = f1_score(y_test, y_pred_test, average='macro')
print(f'F1 score en prueba: {f1:.2f}')

# Calcular el recall del modelo en el conjunto de prueba
recall = recall_score(y_test, y_pred_test, average='macro')
print(f'Recall en prueba: {recall:.2f}')


# Graficar el dataset iris usando las dos primeras características
colors = ['blue', 'red', 'green']
plt.figure(figsize=(8, 6))
for i, class_name in enumerate(iris.target_names):
    plt.scatter(
        X[y == i, 0],  # sepal length
        X[y == i, 1],  # sepal width
        label=class_name,
        color=colors[i],
        edgecolor='black'  # Borde negro para distinguir las muestras
    )
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Iris Dataset (clases reales)')
plt.legend()
plt.show()



# Graficar los datos de prueba coloreados por la clase predicha por el modelo y frontera de decisión
import numpy as np
from matplotlib.colors import ListedColormap
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sin entrenar
for i, class_name in enumerate(iris.target_names):
    axes[0].scatter(
        X[y == i, 0],
        X[y == i, 1],
        label=class_name,
        color=colors[i],
        edgecolor='black'
    )
axes[0].set_xlabel('Sepal length (cm)')
axes[0].set_ylabel('Sepal width (cm)')
axes[0].set_title('Iris Dataset (clases reales, 2 features)')
axes[0].legend()

# Frontera de decisión y datos de prueba
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = knn.predict(grid)
Z = Z.reshape(xx.shape)
cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA', '#AAFFAA'])
axes[1].contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

for i, class_name in enumerate(iris.target_names):
    axes[1].scatter(
        X_test[y_pred_test == i, 0],
        X_test[y_pred_test == i, 1],
        label=f'Predicho: {class_name}',
        color=colors[i],
        edgecolor='black'
    )
axes[1].set_xlabel('Sepal length (cm)')
axes[1].set_ylabel('Sepal width (cm)')
axes[1].set_title('Predicción y frontera KNN (2 features)')
axes[1].legend()

plt.tight_layout()
plt.show()

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred_test)
print("Matriz de confusión:")
print(cm)

# Graficar la matriz de confusión
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names).plot(cmap=plt.cm.Blues)
plt.title('Matriz de confusión del modelo KNN')
plt.show()


