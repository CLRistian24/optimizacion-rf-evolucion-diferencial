import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# *** CARGA ***
df = pd.read_csv("Archivos/dataset.csv")

# *** VARIABLES ***
TARGET = "Class"   # ajusta si tu variable objetivo tiene otro nombre

# *** SEPARACION INICIAL ***
X = df.drop(columns=[TARGET])
y = df[TARGET]

# =========================================================
# *** DIVISION ESTRATIFICADA 70 / 15 / 15 ***
# =========================================================

# 70% train, 30% temporal
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# dividir el 30% en 15% val y 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

# =========================================================
# *** NORMALIZACION DE AMOUNT (SOLO CON TRAIN) ***
# =========================================================

if "Amount" in X_train.columns:
    
    # OPCION 1: Z-SCORE
    scaler = StandardScaler()
    
    # OPCION 2 (alternativa): MinMax
    # scaler = MinMaxScaler()
    
    # AJUSTE SOLO CON TRAIN
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    
    # TRANSFORMACION CON MISMO SCALER
    X_val["Amount"] = scaler.transform(X_val[["Amount"]])
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])

# =========================================================
# *** TRATAMIENTO DE TIME ***
# =========================================================

if "Time" in X_train.columns:
    
    # OPCION 1: ELIMINAR DIRECTAMENTE
    eliminar_time = True   # cambia a False si quieres evaluarla
    
    if eliminar_time:
        X_train = X_train.drop(columns=["Time"])
        X_val = X_val.drop(columns=["Time"])
        X_test = X_test.drop(columns=["Time"])
    
    else:
        # OPCION 2: TRANSFORMACION CICLICA (mejor que dejarla cruda)
        for df_temp in [X_train, X_val, X_test]:
            df_temp["Time_sin"] = np.sin(2 * np.pi * df_temp["Time"] / df_temp["Time"].max())
            df_temp["Time_cos"] = np.cos(2 * np.pi * df_temp["Time"] / df_temp["Time"].max())
            df_temp.drop(columns=["Time"], inplace=True)

# =========================================================
# *** GUARDADO ***
# =========================================================

X_train.to_csv("X_train.csv", index=False)
X_val.to_csv("X_val.csv", index=False)
X_test.to_csv("X_test.csv", index=False)

y_train.to_csv("y_train.csv", index=False)
y_val.to_csv("y_val.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# =========================================================
# *** REPORTE ***
# =========================================================

print("=== DIVISION COMPLETADA ===")
print("Train:", len(X_train))
print("Validacion:", len(X_val))
print("Test:", len(X_test))

print("\nDistribucion de clases:")
print("Train:\n", y_train.value_counts(normalize=True))
print("Val:\n", y_val.value_counts(normalize=True))
print("Test:\n", y_test.value_counts(normalize=True))
