import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

#загрузка обработанного датасета из первой лабораторной работы (файл "processed_dataset.csv")
df = pd.read_csv("processed_dataset.csv")
print("Первые 5 строк обработанного датасета:")
print(df.head(), "\n")

#если столбец "Transported" имеет тип bool, приведём его к числовому виду (0 и 1)
if df["Transported"].dtype == bool:
    df["Transported"] = df["Transported"].astype(int)

#формірованіе переменных
X_cls = df.drop(columns=["Transported"])
y_cls = df["Transported"]

#разбиеніе датасета на обучающую и тестовую выборки (30% на тест) c учетом пропорций по классам
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
    X_cls, y_cls, test_size=0.3, random_state=42, stratify=y_cls
)

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_cls_train, y_cls_train)
y_rfc_pred = rfc.predict(X_cls_test)

#вычісленіе  метрики для RandomForestClassifier
precision_rfc = precision_score(y_cls_test, y_rfc_pred)
recall_rfc = recall_score(y_cls_test, y_rfc_pred)
f1_rfc = f1_score(y_cls_test, y_rfc_pred)

print("=== RandomForestClassifier Результаты ===")
print(f"Precision: {precision_rfc:.4f}")
print(f"Recall:    {recall_rfc:.4f}")
print(f"F1 Score:  {f1_rfc:.4f}")
print("Classification Report:")
print(classification_report(y_cls_test, y_rfc_pred))
print("\n")

gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_cls_train, y_cls_train)
y_gbc_pred = gbc.predict(X_cls_test)

# метрікі для GradientBoostingClassifier
precision_gbc = precision_score(y_cls_test, y_gbc_pred)
recall_gbc = recall_score(y_cls_test, y_gbc_pred)
f1_gbc = f1_score(y_cls_test, y_gbc_pred)

print("=== GradientBoostingClassifier Результаты ===")
print(f"Precision: {precision_gbc:.4f}")
print(f"Recall:    {recall_gbc:.4f}")
print(f"F1 Score:  {f1_gbc:.4f}")
print("Classification Report:")
print(classification_report(y_cls_test, y_gbc_pred))
print("\n")

#компаре
print("=== Сравнение моделей по метрикам ===")
print(f"RandomForestClassifier: Precision = {precision_rfc:.4f}, Recall = {recall_rfc:.4f}, F1 = {f1_rfc:.4f}")
print(f"GradientBoostingClassifier: Precision = {precision_gbc:.4f}, Recall = {recall_gbc:.4f}, F1 = {f1_gbc:.4f}")


#Если одна модель показывает лучшие значения по ключевым метрикам,
#стоит сфокусироваться на её гиперпараметрах и, при необходимости, провести подстройку (GridSearchCV/RandomizedSearchCV).
