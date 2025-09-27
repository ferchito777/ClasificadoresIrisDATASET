"""
¿Cómo funcionan LinearSVC, SVC y NuSVC?

LinearSVC:
Implementa un clasificador de vectores de soporte (SVM) lineal.
Busca el hiperplano que mejor separa las clases en el espacio de características usando una función lineal.
Es eficiente para grandes conjuntos de datos y problemas linealmente separables.

SVC:
Es un clasificador SVM que permite usar diferentes funciones kernel (lineal, polinomial, RBF, etc.),
lo que lo hace adecuado para problemas no lineales.
Busca el hiperplano óptimo en un espacio transformado por el kernel.

NuSVC:
Variante de SVC donde el parámetro de regularización se controla con 'nu' en vez de 'C'.
Ofrece una forma alternativa de ajustar el número de soporte y el margen, útil en ciertos casos de clasificación.
"""
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC, SVC, NuSVC


iris = load_iris()
# Usar solo las dos primeras características para visualización y entrenamiento
X = iris.data[:, :2]
y = iris.target

# Gráfica simple del dataset iris (primeras dos características)
plt.figure(figsize=(8, 6))
for i, color, label in zip(range(3), ['red', 'green', 'blue'], iris.target_names):
	plt.scatter(X[y == i, 0], X[y == i, 1], color=color, label=label)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Iris dataset (sin entrenar)')
plt.legend()
plt.show()


linear_svc = LinearSVC()
svc = SVC()
nu_svc = NuSVC()

# Entrenamiento solo con las dos primeras características
linear_svc.fit(X, y)
svc.fit(X, y)
nu_svc.fit(X, y)

# Calcular accuracy para cada clasificador
linear_acc = accuracy_score(y, linear_svc.predict(X))
svc_acc = accuracy_score(y, svc.predict(X))
nu_svc_acc = accuracy_score(y, nu_svc.predict(X))

print(f"LinearSVC accuracy: {linear_acc:.4f}")
print(f"SVC accuracy: {svc_acc:.4f}")
print(f"NuSVC accuracy: {nu_svc_acc:.4f}")

# Calcular precision para cada clasificador
linear_prec = precision_score(y, linear_svc.predict(X), average='macro')
svc_prec = precision_score(y, svc.predict(X), average='macro')
nu_svc_prec = precision_score(y, nu_svc.predict(X), average='macro')

print(f"LinearSVC precision: {linear_prec:.4f}")
print(f"SVC precision: {svc_prec:.4f}")
print(f"NuSVC precision: {nu_svc_prec:.4f}")

# Calcular f1-score para cada clasificador
linear_f1 = f1_score(y, linear_svc.predict(X), average='macro')
svc_f1 = f1_score(y, svc.predict(X), average='macro')
nu_svc_f1 = f1_score(y, nu_svc.predict(X), average='macro')

print(f"LinearSVC f1-score: {linear_f1:.4f}")
print(f"SVC f1-score: {svc_f1:.4f}")
print(f"NuSVC f1-score: {nu_svc_f1:.4f}")

# Calcular recall para cada clasificador
linear_recall = recall_score(y, linear_svc.predict(X), average='macro')
svc_recall = recall_score(y, svc.predict(X), average='macro')
nu_svc_recall = recall_score(y, nu_svc.predict(X), average='macro')


print(f"LinearSVC recall: {linear_recall:.4f}")
print(f"SVC recall: {svc_recall:.4f}")
print(f"NuSVC recall: {nu_svc_recall:.4f}")


# Gráficas de los datos clasificados por cada modelo en una sola ventana
modelos = [
	(linear_svc, 'LinearSVC'),
	(svc, 'SVC'),
	(nu_svc, 'NuSVC')
]

# Graficar los datos clasificados por cada modelo en subplots con fronteras de decisión
import numpy as np
from matplotlib.colors import ListedColormap
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, (modelo, nombre) in zip(axes, modelos):
	# Fronteras de decisión
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
	grid = np.c_[xx.ravel(), yy.ravel()]
	Z = modelo.predict(grid)
	Z = Z.reshape(xx.shape)
	cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
	ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

	y_pred = modelo.predict(X)
	for i, color, label in zip(range(3), ['red', 'green', 'blue'], iris.target_names):
		ax.scatter(X[y_pred == i, 0], X[y_pred == i, 1], color=color, label=label)
	ax.set_xlabel(iris.feature_names[0])
	ax.set_ylabel(iris.feature_names[1])
	ax.set_title(f'{nombre} (con frontera)')
	ax.legend()
fig.suptitle('Iris dataset clasificado por cada modelo con frontera de decisión')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Matriz de confusión para cada clasificador
print("\nMatriz de confusión LinearSVC:")
print(confusion_matrix(y, linear_svc.predict(X)))
print("\nMatriz de confusión SVC:")
print(confusion_matrix(y, svc.predict(X)))
print("\nMatriz de confusión NuSVC:")
print(confusion_matrix(y, nu_svc.predict(X)))


# Visualización de las tres matrices de confusión en una sola ventana
matrices = [
	(confusion_matrix(y, linear_svc.predict(X)), 'LinearSVC'),
	(confusion_matrix(y, svc.predict(X)), 'SVC'),
	(confusion_matrix(y, nu_svc.predict(X)), 'NuSVC')
]
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, (matriz, nombre) in zip(axes, matrices):
	sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax)
	ax.set_xlabel('Predicción')
	ax.set_ylabel('Real')
	ax.set_title(f'Matriz de confusión - {nombre}')
fig.suptitle('Matrices de confusión de los clasificadores')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
