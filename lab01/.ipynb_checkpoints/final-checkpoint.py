import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("train.csv")
print("Первые 5 строк исходного датасета:")
print(df.head(), "\n")

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Заполнили пропуски в числовом столбце '{col}' медианой: {median_val}")

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"Заполнили пропуски в категориальном столбце '{col}' модой: {mode_val}")

print("\nПосле заполнения пропусков:")
print(df.isnull().sum())


scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("\nНормализованные числовые данные (первые 5 строк):")
print(df[numeric_cols].head())

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("\nДатасет после кодирования категориальных столбцов (первые 5 строк):")
print(df.head())


df.to_csv("processed_dataset.csv", index=False)
print("\nОбработанный датасет сохранён в 'processed_dataset.csv'")
