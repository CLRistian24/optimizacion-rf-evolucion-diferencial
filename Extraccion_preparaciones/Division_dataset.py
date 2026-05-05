import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# CARGA
df = pd.read_csv("Archivos/muestra_limpia.csv")

TARGET = "Class"

# SEPARAR POR CLASE
df_0 = df[df[TARGET] == 0]
df_1 = df[df[TARGET] == 1]

# FUNCION DE DIVISION 70/15/15
def dividir(df_clase):
    n = len(df_clase)
    
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    
    df_clase = df_clase.sample(frac=1, random_state=42)  # shuffle
    
    train = df_clase.iloc[:train_end]
    val = df_clase.iloc[train_end:val_end]
    test = df_clase.iloc[val_end:]
    
    return train, val, test

# DIVIDIR CADA CLASE
train_0, val_0, test_0 = dividir(df_0)
train_1, val_1, test_1 = dividir(df_1)

# COMBINAR
train = pd.concat([train_0, train_1]).sample(frac=1, random_state=42)
val = pd.concat([val_0, val_1]).sample(frac=1, random_state=42)
test = pd.concat([test_0, test_1]).sample(frac=1, random_state=42)

# SEPARAR X Y
X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]

X_val = val.drop(columns=[TARGET])
y_val = val[TARGET]

X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]

# NORMALIZACION
if "Amount" in X_train.columns:
    scaler = StandardScaler()
    
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_val["Amount"] = scaler.transform(X_val[["Amount"]])
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])

# TIME
if "Time" in X_train.columns:
    X_train = X_train.drop(columns=["Time"])
    X_val = X_val.drop(columns=["Time"])
    X_test = X_test.drop(columns=["Time"])

# GUARDADO
X_train.to_csv("X_train.csv", index=False)
X_val.to_csv("X_val.csv", index=False)
X_test.to_csv("X_test.csv", index=False)

y_train.to_csv("y_train.csv", index=False)
y_val.to_csv("y_val.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# REPORTE
print("=== DIVISION COMPLETADA ===")
print("Train:", len(X_train))
print("Val:", len(X_val))
print("Test:", len(X_test))

print("\nDistribucion de clases:")
print("Train:\n", y_train.value_counts(normalize=True))
print("Val:\n", y_val.value_counts(normalize=True))
print("Test:\n", y_test.value_counts(normalize=True))

print("\nConteo absoluto:")
print("Train:\n", y_train.value_counts())
print("Val:\n", y_val.value_counts())
print("Test:\n", y_test.value_counts())