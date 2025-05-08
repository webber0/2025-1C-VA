from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump, load
import numpy as np
import matplotlib.pyplot as plt
# 1. Carga de los descriptores y las etiquetas
X = np.array([
    [0.00065361, 0.00000000, 0.00000000, 0.00000000, -0.00000000, 0.00000000, -0.00000000],
    [0.00000000, 0.00000000, 0.00000000, -0.00000000, 0.00000000, -0.00000000, 0.00065361],
    [0.00065361, 0.00000000, 0.00000000, 0.00000000, -0.00000000, 0.00000000, -0.00000000],
    [0.00000000, -0.00000000, 0.00000000, -0.00000000, 0.00065361, 0.00000000, 0.00000000],
    [0.00000000, -0.00000000, 0.00065361, 0.00000000, 0.00000000, 0.00000000, -0.00000000],
    [0.00065361, 0.00000000, 0.00000000, 0.00000000, -0.00000000, 0.00000000, -0.00000000],
    [0.00000000, 0.00000000, -0.00000000, 0.00000000, -0.00000000, 0.00065361, 0.00000000]
])
Y = np.array([1, 1, 1, 2, 2, 3, 3])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
stratify=Y

# 3. Inicialización y entrenamiento del clasificador
clasificador = tree.DecisionTreeClassifier(random_state=42).fit(X_train, y_train)


# 4. Predicción sobre el conjunto de prueba
y_pred = clasificador.predict(X_test)

# 5. Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo en el conjunto de prueba: {accuracy:.2f}")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))


# 6. Visualización del árbol de decisión
tree.plot_tree(clasificador, filled=True, class_names=[str(i) for i in np.unique(Y)])
plt.show()

# 7. Guardado del modelo entrenado
#nombre_archivo = 'modelo_entrenado.joblib'
#dump(clasificador, nombre_archivo)
#print(f"\nModelo guardado en: {nombre_archivo}")

# 8. Esto para lo que es el clasificador
# clasificador_cargado = load('modelo_entrenado.joblib')
# y luego usarías clasificador_cargado.predict(nuevos_descriptores) para clasificar nuevas formas.