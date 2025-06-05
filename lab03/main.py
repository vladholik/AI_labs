import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
import matplotlib.pyplot as plt

#апрувни через первую лабу csv файл из final.py
df = pd.read_csv("processed_dataset.csv")
print("Первые 5 строк обработанного датасета:")
print(df.head(), "\n")

#если в датасете нет 'Age', выбіраем другой непрерывный признак
if "Age" not in df.columns:
    raise ValueError("В датасете нет столбца 'Age', выберите другой непрерывный признак для регрессии.")

X_reg = df.drop(columns=["Age", "Transported"])
y_reg = df["Age"]

#делим выборку на обучающую и тестовую (30% тест) с фиксированным random_state для воспроизводимости
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42)

#обученіе моделі
regressor = LinearRegression()
regressor.fit(X_reg_train, y_reg_train)

#предікты і кволіті
y_reg_pred = regressor.predict(X_reg_test)
mse = mean_squared_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print("=== Задача регрессии (предсказание Age) ===")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R^2 Score: {r2:.4f}\n")

#далее все аналогічно
if df["Transported"].dtype == bool:
    df["Transported"] = df["Transported"].astype(int)

X_cls = df.drop(columns=["Transported"])  # признаки – все столбцы, кроме целевого
y_cls = df["Transported"]

X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
    X_cls, y_cls, test_size=0.3, random_state=42, stratify=y_cls)

#обученіе классіфікаціі
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_cls_train, y_cls_train)

#получение предсказанных вероятностей для положительного класса (обычно класс "1")
y_prob = classifier.predict_proba(X_cls_test)[:, 1]

#вычисляем ROC-кривую и AUC
fpr, tpr, thresholds = roc_curve(y_cls_test, y_prob)
roc_auc = auc(fpr, tpr)

print("=== Задача классификации (предсказание Transported) ===")
print(f"ROC AUC: {roc_auc:.4f}\n")

#построение ROC-кривой
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
