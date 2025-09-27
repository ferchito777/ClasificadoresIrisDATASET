
"""
¿Cómo funciona RandomForestClassifier?
Random Forest es un método de aprendizaje conjunto que construye múltiples árboles de decisión durante el entrenamiento y predice la clase más frecuente (clasificación) o el promedio (regresión) de los árboles.
Cada árbol se entrena con una muestra aleatoria de los datos y selecciona aleatoriamente las características en cada división, lo que reduce el sobreajuste y mejora la generalización.
Es robusto ante ruido y funciona bien con datos de alta dimensión.
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

iris = load_iris()




# Usar solo las dos primeras características para visualización y entrenamiento
X = iris.data[:, :2]
y = iris.target




# Instancias de los clasificadores
rf_clf = RandomForestClassifier()
ada_clf = AdaBoostClassifier()
gb_clf = GradientBoostingClassifier()

# Entrenamiento solo con las dos primeras características
rf_clf.fit(X, y)
ada_clf.fit(X, y)
gb_clf.fit(X, y)


# Evaluar los modelos con métricas
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
modelos = [
	(rf_clf, "RandomForest"),
	(ada_clf, "AdaBoost"),
	(gb_clf, "GradientBoosting")
]
for modelo, nombre in modelos:
	y_pred = modelo.predict(X)
	acc = accuracy_score(y, y_pred)
	prec = precision_score(y, y_pred, average='macro')
	f1 = f1_score(y, y_pred, average='macro')
	rec = recall_score(y, y_pred, average='macro')
	print(f"\n{nombre} Results:")
	print(f"Accuracy: {acc:.4f}")
	print(f"Precision: {prec:.4f}")
	print(f"F1 Score: {f1:.4f}")
	print(f"Recall: {rec:.4f}")



# Graficar todas las visualizaciones en una sola ventana
fig, axes = plt.subplots(2, 4, figsize=(24, 12))


# Sin entrenar
for i, color, label in zip(range(3), ['red', 'green', 'blue'], iris.target_names):
	axes[0, 0].scatter(X[y == i, 0], X[y == i, 1], color=color, label=label)
axes[0, 0].set_xlabel(iris.feature_names[0])
axes[0, 0].set_ylabel(iris.feature_names[1])
axes[0, 0].set_title('Iris dataset (sin entrenar, 2 features)')
axes[0, 0].legend()



# Clasificado por cada modelo con fronteras de decisión
import numpy as np
from matplotlib.colors import ListedColormap
for idx, (modelo, nombre) in enumerate(modelos, start=1):
	# Fronteras de decisión
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
	grid = np.c_[xx.ravel(), yy.ravel()]
	Z = modelo.predict(grid)
	Z = Z.reshape(xx.shape)
	cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
	axes[0, idx].contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

	y_pred = modelo.predict(X)
	for i, color, label in zip(range(3), ['red', 'green', 'blue'], iris.target_names):
		axes[0, idx].scatter(X[y_pred == i, 0], X[y_pred == i, 1], color=color, label=label)
	axes[0, idx].set_xlabel(iris.feature_names[0])
	axes[0, idx].set_ylabel(iris.feature_names[1])
	axes[0, idx].set_title(f'Iris clasificado por {nombre} (2 features)')
	axes[0, idx].legend()

	# Matriz de confusión
	cm = confusion_matrix(y, y_pred)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
	disp.plot(ax=axes[1, idx], cmap='Blues', colorbar=False)
	axes[1, idx].set_title(f'Matriz de confusión {nombre}')

# Título y matriz vacía para el primer subplot de la segunda fila
axes[1, 0].axis('off')
axes[1, 0].set_title('')

plt.tight_layout()
plt.show()

