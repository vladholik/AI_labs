import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

#апрувни через первую лабу csv файл из final.py
df = pd.read_csv("processed_dataset.csv")
print("Первые 5 строк обработанного датасета:")
print(df.head(), "\n")

if "Age" not in df.columns:
    raise ValueError("В датасете нет столбца 'Age', выберите другой непрерывный признак для регрессии.")

#переменные для регрессии
X_reg = df.drop(columns=["Age", "Transported"])  # все столбцы, кроме 'Age' и 'Transported'
y_reg = df["Age"]

#выборка на обуч и тест с 30% текста
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

#модели линейной регрессии
regressor = LinearRegression()
regressor.fit(X_reg_train, y_reg_train)

#предсказывание по текстовой выборке
y_reg_pred = regressor.predict(X_reg_test)
mse = mean_squared_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print("=== Задача регрессии (предсказание Age) ===")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

#если качество модели невысокое (например, R^2 ниже 0.5), печатаем предложения по улучшению
if r2 < 0.5:
    print("\nРезультаты модели регрессии неудовлетворительны. Возможные пути улучшения:")
    print(" - Попробуйте использовать более сложные модели, например, RandomForestRegressor или GradientBoostingRegressor.")
    print(" - Проведите дополнительный отбор признаков (feature selection) или создание новых признаков.")
    print(" - Поэксперементируйте с гиперпараметрами модели, применив GridSearchCV или RandomizedSearchCV.\n")

#классификация целевого признака – "Transported" если он имеет тип bool, приведём к целочисленному виду (0 и 1)
if df["Transported"].dtype == bool:
    df["Transported"] = df["Transported"].astype(int)

# формирование переменных
X_cls = df.drop(columns=["Transported"])  # удаляем только целевой столбец
y_cls = df["Transported"]

X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
    X_cls, y_cls, test_size=0.3, random_state=42, stratify=y_cls
)

#модель логической регрессии
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_cls_train, y_cls_train)

#предсказывание классов
y_cls_pred = classifier.predict(X_cls_test)
accuracy = accuracy_score(y_cls_test, y_cls_pred)

print("=== Задача классификации (предсказание Transported) ===")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_cls_test, y_cls_pred))

#если точность меньше 70% предложения по улучшению модели
if accuracy < 0.7:
    print("\nРезультаты модели классификации неудовлетворительны. Возможные пути улучшения:")
    print(" - Используйте более сложные модели, такие как RandomForestClassifier или GradientBoostingClassifier.")
    print(" - Проведите подбор гиперпараметров с помощью GridSearchCV или RandomizedSearchCV.")
    print(" - Если классы несбалансированы, примените методы балансировки (например, oversampling, undersampling или использование класса weights).")
