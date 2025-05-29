from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

CSV_FILE = "dataset_letters.csv"
MODEL_FILE = "letter_model_tp3.pkl"

try:
    data = pd.read_csv(CSV_FILE)
    print(f'File loaded successfully.')
except FileNotFoundError:
    print(f'Error: the file was not found')
    exit()
except Exception as e:
    print(f'An error happen: {e}')
    exit()

labels = data.iloc[:, 0]
features = data.iloc[:, 1:]

expected_features = 21 * 3 #21 landmarks
if features.shape[1] != expected_features:
    print(f'Something went wrong. {features.shape[1]} features found.')

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state = 42)

model = LogisticRegression(max_iter = 1000, solver = 'liblinear')
print('Training model...')
model.fit(x_train, y_train)
print('Model trained.')

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')

try:
    joblib.dump(model, MODEL_FILE)
    print(f'Saved model in: {MODEL_FILE}')
except Exception as e:
    print(f'Error: {e}')