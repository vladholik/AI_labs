import pandas as pd
#для игнора ошибок связанных с версией пандаса
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


df = pd.read_csv("train.csv")

#списки столбцов по типам
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

#заполнение пропусков для числовых столбцов медианой
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Заполнили пропуски в числовом столбце '{col}' медианой: {median_val}")

#заполнение пропусков для категориальных столбиков модой
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"Заполнили пропуски в категориальном столбце '{col}' модой: {mode_val}")

print("\nПосле заполнения пропусков:")
print(df.isnull().sum())
