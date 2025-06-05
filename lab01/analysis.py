import  pandas as pd

df = pd.read_csv("train.csv")

print("5 первых строк датасета")
print(df.head(), "\n")

print("Пропущенные значения по столбцам")
print(df.isnull().sum())