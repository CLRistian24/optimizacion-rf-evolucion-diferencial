import pandas as pd
import numpy as np

# CARGA DE DATOS 
df = pd.read_csv("Archivos/creditcard.csv")

# PARAMETROS 
TAM_BLOQUE = 5000

# FUNCION DE LIMPIEZA POR BLOQUE
def limpiar_bloque(bloque):
    
    # DETECTAR NULOS
    if bloque.isnull().values.any():
        
        # IMPUTACION POR MEDIANA (SOLO COLUMNAS NUMERICAS)
        for col in bloque.select_dtypes(include=[np.number]).columns:
            mediana = bloque[col].median()
            bloque[col] = bloque[col].fillna(mediana)
    
    # CONTROL DE VALORES EXTREMOS 
    for col in bloque.select_dtypes(include=[np.number]).columns:
        q1 = bloque[col].quantile(0.25)
        q3 = bloque[col].quantile(0.75)
        iqr = q3 - q1
        
        limite_inf = q1 - 1.5 * iqr
        limite_sup = q3 + 1.5 * iqr
        
        bloque[col] = np.clip(bloque[col], limite_inf, limite_sup)
    
    return bloque

# PROCESAMIENTO POR BLOQUES 
bloques_limpios = []

for i in range(0, len(df), TAM_BLOQUE):
    bloque = df.iloc[i:i+TAM_BLOQUE].copy()
    bloque_limpio = limpiar_bloque(bloque)
    bloques_limpios.append(bloque_limpio)

# RECONSTRUCCION DEL DATASET
df_limpio = pd.concat(bloques_limpios, ignore_index=True)

# GUARDADO 
df_limpio.to_csv("muestra_limpia.csv", index=False)

# REPORTE BASICO 
print("Proceso terminado")
print("Valores nulos finales:")
print(df_limpio.isnull().sum())