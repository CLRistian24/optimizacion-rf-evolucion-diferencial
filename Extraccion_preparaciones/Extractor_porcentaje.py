#!/usr/bin/env python3
"""
Script para extraer un porcentaje aleatorio de filas de un CSV.
CONFIGURACIÓN DIRECTA: edita las rutas y opciones abajo.
"""

import pandas as pd
import sys

# ========== CONFIGURACIÓN DIRECTA ==========
ARCHIVO_ENTRADA = r"creditcard.csv"   # Ruta completa del archivo original
ARCHIVO_SALIDA  = r"muestra.csv" # Ruta donde guardar la muestra
PORCENTAJE = 5.0           # Porcentaje a extraer (ej: 5 para 5%)
SEMILLA = 42             # Opcional: número fijo para reproducibilidad (ej: 42)
DELIMITADOR = ","          # Separador del CSV (',' o ';' etc.)
CODIFICACION = "utf-8"     # Codificación del archivo (utf-8, latin1, etc.)
# ===========================================

def main():
    # Validar porcentaje
    if PORCENTAJE <= 0 or PORCENTAJE > 100:
        print("Error: El porcentaje debe estar entre 0 y 100.", file=sys.stderr)
        sys.exit(1)

    try:
        # Leer el CSV
        print(f"Leyendo {ARCHIVO_ENTRADA} ...")
        df = pd.read_csv(ARCHIVO_ENTRADA, encoding=CODIFICACION, delimiter=DELIMITADOR)

        total_filas = len(df)
        if total_filas == 0:
            print("El archivo CSV está vacío.", file=sys.stderr)
            sys.exit(1)

        # Calcular tamaño de la muestra
        frac = PORCENTAJE / 100.0
        n_muestra = int(round(frac * total_filas))

        if n_muestra == 0:
            print(f"Advertencia: {PORCENTAJE}% de {total_filas} filas es 0. No se guardará nada.")
            sys.exit(0)

        # Fijar semilla si se proporciona
        if SEMILLA is not None:
            import random
            import numpy as np
            random.seed(SEMILLA)
            np.random.seed(SEMILLA)

        # Tomar muestra aleatoria
        print(f"Extrayendo {n_muestra} filas ({PORCENTAJE}%) de {total_filas} ...")
        muestra = df.sample(n=n_muestra, random_state=SEMILLA)

        # Guardar resultado
        muestra.to_csv(ARCHIVO_SALIDA, index=False, encoding=CODIFICACION)
        print(f"Muestra guardada en {ARCHIVO_SALIDA}")

    except FileNotFoundError:
        print(f"Error: No se encuentra el archivo {ARCHIVO_ENTRADA}", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: El archivo {ARCHIVO_ENTRADA} está vacío o no es un CSV válido.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error inesperado: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()