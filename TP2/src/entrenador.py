import pandas as pd # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import classification_report # type: ignore
import joblib # type: ignore

# -------------------- CARGA DEL DATASET --------------------
df = pd.read_csv('hu_dataset.csv', header = 0)

# Se separan características (X) y etiquetas (y)
X = df.iloc[:, :-1]  # Todas las columnas menos la última
y = df.iloc[:, -1]   # Última columna = etiqueta

# print(df.head())
# print(df.dtypes)

# -------------------- DIVISIÓN ENTRENAMIENTO/TEST --------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- ENTRENAMIENTO DEL MODELO --------------------

modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

# -------------------- EVALUACIÓN --------------------

y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# -------------------- GUARDAR MODELO ENTRENADO --------------------
joblib.dump(modelo, 'modelo_entrenado.joblib')
print('Modelo guardado como modelo_entrenado.joblib')