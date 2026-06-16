import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, recall_score, roc_auc_score,
                              average_precision_score)


RUTA_X_TRAIN = "Archivos/Division/X_train.csv" 
RUTA_Y_TRAIN = "Archivos/Division/y_train.csv"
RUTA_X_VAL   = "Archivos/Division/X_val.csv"
RUTA_Y_VAL   = "Archivos/Division/y_val.csv"
RUTA_X_TEST  = "Archivos/Division/X_test.csv"
RUTA_Y_TEST  = "Archivos/Division/y_test.csv"

CARPETA_SALIDA = "Resultados_DE_RF_Checkpoint"

# Parametros de la Evolucion Diferencial
NP       = 6
F        = 0.8
CR       = 0.9
MAX_GEN  = 10
PACIENCIA = 2
TOL      = 1e-6
SEMILLA  = NULL

# Numero de procesos para la version paralela (nucleos disponibles menos uno)
N_WORKERS = max(1, mp.cpu_count() - 1)

# ESPACIO DE BUSQUEDA DE HIPERPARAMETROS (grilla actualizada)

ESPACIO = {
    "n_estimators":      (10.0,  80.0),
    "max_depth":         (3.0,   40.0),
    "min_samples_split": (2.0,   40.0),
    "min_samples_leaf":  (1.0,   15.0),
    "max_features":      (0.15,  0.9),
    "bootstrap":         (0.0,   1.0),
    "class_weight":      (0.0,   2.0),   # 0->balanced  1->None  2->balanced_subsample
    "criterion":         (0.0,   1.0),   # 0->entropy   1->gini
}

NOMBRES  = list(ESPACIO.keys())
N_DIMS   = len(NOMBRES)
LIM_INF  = np.array([ESPACIO[k][0] for k in NOMBRES])
LIM_SUP  = np.array([ESPACIO[k][1] for k in NOMBRES])

# Mapeos discretos alineados con la grilla pedida
DECODE_BOOTSTRAP     = [False, True]
DECODE_CLASS_WEIGHT  = ["balanced", None, "balanced_subsample"]
DECODE_CRITERION     = ["entropy", "gini"]


def decodificar(v):
    """Convierte un vector continuo en hiperparametros validos para RandomForestClassifier."""
    return {
        "n_estimators":      int(round(v[0])),
        "max_depth":         int(round(v[1])),
        "min_samples_split": int(round(v[2])),
        "min_samples_leaf":  int(round(v[3])),
        "max_features":      float(np.clip(v[4], 0.15, 0.9)),
        "bootstrap":         DECODE_BOOTSTRAP[int(round(np.clip(v[5], 0, 1)))],
        "class_weight":      DECODE_CLASS_WEIGHT[int(round(np.clip(v[6], 0, 2)))],
        "criterion":         DECODE_CRITERION[int(round(np.clip(v[7], 0, 1)))],
    }


# NIVELACION DE CARGAS

def nivelacion_cargas(n_p, indices):
    """Reparte una lista de indices entre n_p procesos de forma equitativa."""
    n_p = min(n_p, len(indices))
    if n_p <= 0:
        return [indices]
    s   = len(indices) % n_p
    n_d = indices[:s]
    t   = (len(indices) - s) // n_p
    out, temp = [], []
    for i in indices[s:]:
        temp.append(i)
        if len(temp) == t:
            out.append(temp)
            temp = []
    for i in range(len(n_d)):
        out[i].append(n_d[i])
    return out


# EVALUACION DE UN INDIVIDUO

def evaluar_individuo(v, X_tr_arr, y_tr_arr, X_val_arr, y_val_arr, n_estimators_workers=1):
    params = decodificar(v)
    if n_estimators_workers > 1:
        modelo = entrenar_rf_paralelo(params, X_tr_arr, y_tr_arr, n_estimators_workers)
    else:
        modelo = RandomForestClassifier(random_state=42, **params)
        modelo.fit(X_tr_arr, y_tr_arr)
    y_pred_val = modelo.predict(X_val_arr)
    y_prob_val = modelo.predict_proba(X_val_arr)[:, 1]
    f1  = f1_score(y_val_arr, y_pred_val, pos_label=1, zero_division=0)
    apr = average_precision_score(y_val_arr, y_prob_val)
    return -(0.7 * f1 + 0.3 * apr)


# ENTRENAMIENTO PARALELO DEL BOSQUE ALEATORIO (modo RF)

def _entrenar_subbosque(args):
    n_arboles, params_base, X_tr_arr, y_tr_arr, semilla_sub = args
    params = dict(params_base)
    params["n_estimators"] = n_arboles
    modelo = RandomForestClassifier(random_state=semilla_sub, **params)
    modelo.fit(X_tr_arr, y_tr_arr)
    return modelo.estimators_, modelo.classes_, modelo.n_outputs_, modelo.n_features_in_


def entrenar_rf_paralelo(params, X_tr_arr, y_tr_arr, n_workers):
    n_total      = params["n_estimators"]
    params_sin_n = {k: v for k, v in params.items() if k != "n_estimators"}
    grupos       = nivelacion_cargas(n_workers, list(range(n_total)))
    tareas       = [(len(g), params_sin_n, X_tr_arr, y_tr_arr, 1000 + i)
                    for i, g in enumerate(grupos)]
    with mp.Pool(processes=min(n_workers, len(tareas))) as pool:
        resultados = pool.map(_entrenar_subbosque, tareas)

    arboles_totales = []
    for arboles, *_ in resultados:
        arboles_totales.extend(arboles)

    modelo_final = RandomForestClassifier(random_state=42, **params)
    modelo_final.n_estimators  = len(arboles_totales)
    modelo_final.estimators_   = arboles_totales
    modelo_final.classes_      = resultados[0][1]
    modelo_final.n_classes_    = len(resultados[0][1])
    modelo_final.n_outputs_    = resultados[0][2]
    modelo_final.n_features_in_ = resultados[0][3]
    return modelo_final


# WORKER PARA LA PARALELIZACION EN LA EVOLUCION DIFERENCIAL (modo DE)

def _worker_chunk_de(args_chunk):
    resultados = []
    for idx, v, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr in args_chunk:
        t0  = time.time()
        fit = evaluar_individuo(v, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr, n_estimators_workers=1)
        resultados.append((idx, fit, time.time() - t0))
    return resultados


def evaluar_poblacion_de_paralelo(vectores, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr, n_workers):
    n          = len(vectores)
    chunks_idx = nivelacion_cargas(n_workers, list(range(n)))
    args       = [[(idx, vectores[idx], X_tr_arr, y_tr_arr, X_v_arr, y_v_arr)
                   for idx in chunk] for chunk in chunks_idx]
    with mp.Pool(processes=min(n_workers, len(args))) as pool:
        resultados_chunks = pool.map(_worker_chunk_de, args)
    aptitud, tiempos = [None] * n, []
    for chunk in resultados_chunks:
        for idx, fit, t_eval in chunk:
            aptitud[idx] = fit
            tiempos.append(t_eval)
    return np.array(aptitud), tiempos


def evaluar_poblacion_secuencial(vectores, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr, n_workers_rf):
    aptitud, tiempos = np.zeros(len(vectores)), []
    for i, v in enumerate(vectores):
        t0 = time.time()
        aptitud[i] = evaluar_individuo(v, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr,
                                       n_estimators_workers=n_workers_rf)
        tiempos.append(time.time() - t0)
    return aptitud, tiempos


# FUNCIONES DE CHECKPOINT
# Guarda/carga el estado completo de una corrida para poder reanudarla.

def ruta_checkpoint(carpeta, etiqueta):
    return os.path.join(carpeta, f"checkpoint_{etiqueta}.pkl")


def guardar_checkpoint(carpeta, etiqueta, estado):
    ruta     = ruta_checkpoint(carpeta, etiqueta)
    ruta_tmp = ruta + ".tmp"
    with open(ruta_tmp, "wb") as f:
        pickle.dump(estado, f)
    os.replace(ruta_tmp, ruta)


def cargar_checkpoint(carpeta, etiqueta):
    ruta = ruta_checkpoint(carpeta, etiqueta)
    if os.path.exists(ruta):
        with open(ruta, "rb") as f:
            return pickle.load(f)
    return None


# CICLO PRINCIPAL DE LA EVOLUCION DIFERENCIAL CON CHECKPOINTS

def evolucion_diferencial(NP, F, CR, max_gen, paciencia, tol, semilla,
                           X_train, y_train, X_val, y_val,
                           paralelo, modo, n_workers, carpeta, etiqueta):
    X_tr_arr = X_train.values
    y_tr_arr = y_train.values
    X_v_arr  = X_val.values
    y_v_arr  = y_val.values

    n_workers_rf = n_workers if modo == "RF" else 1

    estado = cargar_checkpoint(carpeta, etiqueta)

    if estado is not None:
        print(f"  Checkpoint encontrado para '{etiqueta}', "
              f"reanudando desde generacion {estado['gen_actual']}")
        rng                     = estado["rng"]
        poblacion               = estado["poblacion"]
        aptitud                 = estado["aptitud"]
        mejor_apt               = estado["mejor_apt"]
        mejor_vec               = estado["mejor_vec"]
        hist_mejor              = estado["hist_mejor"]
        hist_prom               = estado["hist_prom"]
        hist_tiempos            = estado["hist_tiempos"]
        tiempos                 = estado["tiempos_evaluacion"]
        tiempo_acumulado_previo = estado["tiempo_acumulado_previo"]
        gen_inicio              = estado["gen_actual"] + 1
        generacion_parada       = estado["gen_actual"]

        if estado.get("terminado", False):
            return {
                "mejor_vec":         mejor_vec.tolist(),
                "hist_mejor":        hist_mejor,
                "hist_prom":         hist_prom,
                "hist_tiempos":      hist_tiempos,
                "tiempos_evaluacion": tiempos,
                "tiempo_total":      tiempo_acumulado_previo,
                "generacion_parada": generacion_parada,
            }
    else:
        rng       = np.random.default_rng(semilla)
        poblacion = rng.uniform(LIM_INF, LIM_SUP, size=(NP, N_DIMS))
        tiempos   = []

        if paralelo and modo == "DE":
            aptitud, t_init = evaluar_poblacion_de_paralelo(
                poblacion, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr, n_workers)
        else:
            aptitud, t_init = evaluar_poblacion_secuencial(
                poblacion, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr, n_workers_rf)
        tiempos.extend(t_init)

        mejor_idx               = np.argmin(aptitud)
        mejor_apt               = aptitud[mejor_idx]
        mejor_vec               = poblacion[mejor_idx].copy()
        hist_mejor              = [-mejor_apt]
        hist_prom               = [-np.mean(aptitud)]
        hist_tiempos            = [0.0]
        tiempo_acumulado_previo = 0.0
        gen_inicio              = 1
        generacion_parada       = 0

        guardar_checkpoint(carpeta, etiqueta, {
            "rng": rng, "poblacion": poblacion, "aptitud": aptitud,
            "mejor_apt": mejor_apt, "mejor_vec": mejor_vec,
            "hist_mejor": hist_mejor, "hist_prom": hist_prom,
            "hist_tiempos": hist_tiempos, "tiempos_evaluacion": tiempos,
            "tiempo_acumulado_previo": tiempo_acumulado_previo,
            "gen_actual": 0, "terminado": False,
        })

    inicio_total = time.time()

    for gen in range(gen_inicio, max_gen + 1):
        trials = []
        for i in range(NP):
            candidatos = [j for j in range(NP) if j != i]
            a, b, c    = rng.choice(candidatos, size=3, replace=False)
            donante    = poblacion[a] + F * (poblacion[b] - poblacion[c])
            mask       = rng.random(N_DIMS) < CR
            mask[rng.integers(0, N_DIMS)] = True
            trial      = np.clip(np.where(mask, donante, poblacion[i]), LIM_INF, LIM_SUP)
            trials.append(trial)

        if paralelo and modo == "DE":
            apt_trials, t_gen = evaluar_poblacion_de_paralelo(
                trials, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr, n_workers)
        else:
            apt_trials, t_gen = evaluar_poblacion_secuencial(
                trials, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr, n_workers_rf)
        tiempos.extend(t_gen)

        for i in range(NP):
            if apt_trials[i] <= aptitud[i]:
                poblacion[i] = trials[i]
                aptitud[i]   = apt_trials[i]
                if apt_trials[i] < mejor_apt:
                    mejor_apt = apt_trials[i]
                    mejor_vec = trials[i].copy()

        hist_mejor.append(-mejor_apt)
        hist_prom.append(-np.mean(aptitud))
        tiempo_acumulado_previo += time.time() - inicio_total
        hist_tiempos.append(tiempo_acumulado_previo)
        inicio_total      = time.time()
        generacion_parada = gen

        terminado = False
        if gen >= paciencia:
            ventana = hist_mejor[-paciencia:]
            ref     = hist_mejor[-paciencia - 1]
            if max(ventana) - ref < tol:
                terminado = True
        if gen == max_gen:
            terminado = True

        # Checkpoint al final de cada generacion con el historial COMPLETO acumulado
        guardar_checkpoint(carpeta, etiqueta, {
            "rng": rng, "poblacion": poblacion, "aptitud": aptitud,
            "mejor_apt": mejor_apt, "mejor_vec": mejor_vec,
            "hist_mejor": hist_mejor, "hist_prom": hist_prom,
            "hist_tiempos": hist_tiempos, "tiempos_evaluacion": tiempos,
            "tiempo_acumulado_previo": tiempo_acumulado_previo,
            "gen_actual": gen, "terminado": terminado,
        })

        if terminado:
            break

    return {
        "mejor_vec":         mejor_vec.tolist(),
        "hist_mejor":        hist_mejor,
        "hist_prom":         hist_prom,
        "hist_tiempos":      hist_tiempos,
        "tiempos_evaluacion": tiempos,
        "tiempo_total":      tiempo_acumulado_previo,
        "generacion_parada": generacion_parada,
    }


# EVALUACION FINAL DEL MEJOR MODELO EN EL CONJUNTO DE PRUEBA

def evaluar_modelo_final(mejor_vec, X_train_val, y_train_val, X_test, y_test):
    params = decodificar(mejor_vec)
    modelo = RandomForestClassifier(random_state=42, **params)
    modelo.fit(X_train_val, y_train_val)
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    return {
        "f1":     float(f1_score(y_test, y_pred, pos_label=1)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1)),
        "auc":    float(roc_auc_score(y_test, y_prob)),
        "params": params,
    }


# FUNCION PRINCIPAL

def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    X_train = pd.read_csv(RUTA_X_TRAIN)
    y_train = pd.read_csv(RUTA_Y_TRAIN).squeeze()
    X_val   = pd.read_csv(RUTA_X_VAL)
    y_val   = pd.read_csv(RUTA_Y_VAL).squeeze()
    X_test  = pd.read_csv(RUTA_X_TEST)
    y_test  = pd.read_csv(RUTA_Y_TEST).squeeze()

    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    # CONFIGURACIÓN: lista de números de núcleos a probar
    #LISTA_WORKERS = [2, 4, 5, mp.cpu_count()-1]  
    LISTA_WORKERS = [4]
    
    # Resultados acumulados de todas las configuraciones
    todos_resultados = {}

    print("COMPARACIÓN DE RENDIMIENTO CON DISTINTOS NÚMEROS DE NÚCLEOS")
    print(f"NP={NP}, generaciones máximas={MAX_GEN}")
    print(f"Núcleos a probar: {LISTA_WORKERS}")

    # Versión secuencial (solo se ejecuta UNA vez como referencia)
    print("\n[SECUENCIAL] Ejecutando versión base de comparación...")
    t0 = time.time()
    res_seq = evolucion_diferencial(
        NP=NP, F=F, CR=CR, max_gen=MAX_GEN,
        paciencia=PACIENCIA, tol=TOL, semilla=SEMILLA,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        paralelo=False, modo="DE",
        n_workers=1, carpeta=CARPETA_SALIDA,
        etiqueta="secuencial"
    )
    t_seq = res_seq["tiempo_total"]
    print(f"  -> Tiempo secuencial acumulado: {t_seq:.2f} s\n")

    # Guardar resultado secuencial
    todos_resultados["secuencial"] = {
        "tiempo_total": t_seq,
        "resumen": res_seq,
    }

    # Versiones paralelas con distinto número de workers 
    for n_w in LISTA_WORKERS:
        etiqueta_de = f"paralelo_DE_w{n_w}"
        etiqueta_rf = f"paralelo_RF_w{n_w}"

        print(f"\nNúcleos: {n_w} ")

        # Paralelo DE
        print(f"  [DE] Ejecutando con {n_w} workers...")
        t0_de = time.time()
        res_de = evolucion_diferencial(
            NP=NP, F=F, CR=CR, max_gen=MAX_GEN,
            paciencia=PACIENCIA, tol=TOL, semilla=SEMILLA,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            paralelo=True, modo="DE",
            n_workers=n_w, carpeta=CARPETA_SALIDA,
            etiqueta=etiqueta_de
        )
        t_de = res_de["tiempo_total"]
        speedup_de = t_seq / t_de if t_de > 0 else float("nan")
        eficiencia_de = speedup_de / n_w
        print(f"    Tiempo: {t_de:.2f}s | Speedup: {speedup_de:.2f}x | "
              f"Eficiencia: {eficiencia_de*100:.1f}%")

        # Paralelo RF
        print(f"  [RF] Ejecutando con {n_w} workers...")
        t0_rf = time.time()
        res_rf = evolucion_diferencial(
            NP=NP, F=F, CR=CR, max_gen=MAX_GEN,
            paciencia=PACIENCIA, tol=TOL, semilla=SEMILLA,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            paralelo=True, modo="RF",
            n_workers=n_w, carpeta=CARPETA_SALIDA,
            etiqueta=etiqueta_rf
        )
        t_rf = res_rf["tiempo_total"]
        speedup_rf = t_seq / t_rf if t_rf > 0 else float("nan")
        eficiencia_rf = speedup_rf / n_w
        print(f"    Tiempo: {t_rf:.2f}s | Speedup: {speedup_rf:.2f}x | "
              f"Eficiencia: {eficiencia_rf*100:.1f}%")

        # Evaluación final en test
        final_seq   = evaluar_modelo_final(res_seq["mejor_vec"],
                                           X_train_val.values, y_train_val.values,
                                           X_test.values, y_test.values)
        final_de    = evaluar_modelo_final(res_de["mejor_vec"],
                                           X_train_val.values, y_train_val.values,
                                           X_test.values, y_test.values)
        final_rf    = evaluar_modelo_final(res_rf["mejor_vec"],
                                           X_train_val.values, y_train_val.values,
                                           X_test.values, y_test.values)

        todos_resultados[f"workers_{n_w}"] = {
            "n_workers": n_w,
            "secuencial_final": final_seq,
            "paralelo_DE": {
                "tiempo_total": t_de,
                "speedup": speedup_de,
                "eficiencia": eficiencia_de,
                "resumen": res_de,
                "evaluacion_final": final_de,
            },
            "paralelo_RF": {
                "tiempo_total": t_rf,
                "speedup": speedup_rf,
                "eficiencia": eficiencia_rf,
                "resumen": res_rf,
                "evaluacion_final": final_rf,
            },
        }

        # Guardar resultados parciales por si se interrumpe
        ruta_json_temp = os.path.join(CARPETA_SALIDA, "resultados_parciales.json")
        with open(ruta_json_temp, "w", encoding="utf-8") as f:
            json.dump(todos_resultados, f, indent=2, ensure_ascii=False, default=str)

    # GUARDADO FINAL DE TODOS LOS RESULTADOS EN JSON
    resultados_json = {
        "configuracion": {
            "NP": NP, "F": F, "CR": CR, "MAX_GEN": MAX_GEN,
            "PACIENCIA": PACIENCIA, "TOL": TOL, "SEMILLA": SEMILLA,
            "workers_probados": LISTA_WORKERS,
            "cpu_count": mp.cpu_count(),
        },
        "tiempo_secuencial": t_seq,
        "resultados_por_workers": todos_resultados,
    }

    ruta_json = os.path.join(CARPETA_SALIDA, "resultados.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultados_json, f, indent=2, ensure_ascii=False, default=str)

    # RESUMEN FINAL EN PANTALLA
    print("RESUMEN FINAL")
    print(f"{'Workers':<10} {'Speedup DE':<12} {'Eficiencia DE':<14} "
          f"{'Speedup RF':<12} {'Eficiencia RF':<14}")
    print("-" * 62)
    for n_w in LISTA_WORKERS:
        datos = todos_resultados[f"workers_{n_w}"]
        print(f"{n_w:<10} "
              f"{datos['paralelo_DE']['speedup']:<12.2f} "
              f"{datos['paralelo_DE']['eficiencia']*100:<14.1f}% "
              f"{datos['paralelo_RF']['speedup']:<12.2f} "
              f"{datos['paralelo_RF']['eficiencia']*100:<14.1f}%")

    print(f"\nArchivos guardados en '{CARPETA_SALIDA}':")
    print("  resultadosF.json              <- Todos los resultados")
    print("  resultados_parcialesF.json     <- Copia de respaldo")
    print("  checkpoint_*.pkl              <- Checkpoints por cada ejecución")

if __name__ == "__main__":
    main()