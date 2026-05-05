import pandas as pd
import numpy as np

df = pd.read_csv("Archivos/creditcard.csv")

TAM_BLOQUE = 5000
TARGET = "Class"

def limpiar_bloque(bloque):
    
    columnas_numericas = [
        col for col in bloque.select_dtypes(include=[np.number]).columns
        if col != TARGET
    ]
    
    if bloque[columnas_numericas].isnull().values.any():
        for col in columnas_numericas:
            mediana = bloque[col].median()
            bloque[col] = bloque[col].fillna(mediana)
    
    for col in columnas_numericas:
        q1 = bloque[col].quantile(0.25)
        q3 = bloque[col].quantile(0.75)
        iqr = q3 - q1
        
        limite_inf = q1 - 1.5 * iqr
        limite_sup = q3 + 1.5 * iqr
        
        bloque[col] = np.clip(bloque[col], limite_inf, limite_sup)
    
    return bloque

bloques_limpios = []

for i in range(0, len(df), TAM_BLOQUE):
    bloque = df.iloc[i:i+TAM_BLOQUE].copy()
    bloque_limpio = limpiar_bloque(bloque)
    bloques_limpios.append(bloque_limpio)

df_limpio = pd.concat(bloques_limpios, ignore_index=True)

df_limpio.to_csv("Archivos/muestra_limpia.csv", index=False)

print("Proceso terminado")
print("Valores nulos finales:")
print(df_limpio.isnull().sum())

print("\nDistribucion de clase (antes vs despues):")
print("Original:\n", df["Class"].value_counts())
print("Limpio:\n", df_limpio["Class"].value_counts())