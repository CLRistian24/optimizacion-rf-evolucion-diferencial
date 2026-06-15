import os
import time
import json
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, recall_score, roc_auc_score,
                              confusion_matrix, ConfusionMatrixDisplay,
                              average_precision_score)

# CONFIGURACION GENERAL DEL EXPERIMENTO

RUTA_X_TRAIN = "Archivos/Division/X_train.csv"
RUTA_Y_TRAIN = "Archivos/Division/y_train.csv"
RUTA_X_VAL = "Archivos/Division/X_val.csv"
RUTA_Y_VAL = "Archivos/Division/y_val.csv"
RUTA_X_TEST = "Archivos/Division/X_test.csv"
RUTA_Y_TEST = "Archivos/Division/y_test.csv"

CARPETA_SALIDA = "Resultados_DE_RF"

# Parametros de la evolucion diferencial, se usan tanto en secuencial como en paralelo
NP = 5
F = 0.8
CR = 0.9
MAX_GEN = 4
PACIENCIA = 2
TOL = 1e-4
SEMILLA = 42

# Numero de procesos a usar en la version paralela, en ambos modos
N_WORKERS = max(1, mp.cpu_count() - 1)

# Espacio de busqueda de hiperparametros del bosque aleatorio
ESPACIO = {
    "n_estimators": (10.0, 200.0),
    "max_depth": (3.0, 50.0),
    "min_samples_split": (2.0, 50.0),
    "min_samples_leaf": (1.0, 20.0),
    "max_features": (0.1, 1.0),
    "bootstrap": (0.0, 1.0),
    "class_weight": (0.0, 2.0),
    "criterion": (0.0, 2.0),
}
NOMBRES = list(ESPACIO.keys())
N_DIMS = len(NOMBRES)
LIM_INF = np.array([ESPACIO[k][0] for k in NOMBRES])
LIM_SUP = np.array([ESPACIO[k][1] for k in NOMBRES])

DECODE_BOOTSTRAP = [False, True]
DECODE_CLASS_WEIGHT = [None, "balanced", "balanced_subsample"]
DECODE_CRITERION = ["gini", "entropy", "log_loss"]


def decodificar(v):
    # Convierte un vector continuo del espacio de busqueda en hiperparametros validos para RandomForestClassifier
    return {
        "n_estimators": int(round(v[0])),
        "max_depth": int(round(v[1])),
        "min_samples_split": int(round(v[2])),
        "min_samples_leaf": int(round(v[3])),
        "max_features": float(np.clip(v[4], 0.1, 1.0)),
        "bootstrap": DECODE_BOOTSTRAP[int(round(np.clip(v[5], 0, 1)))],
        "class_weight": DECODE_CLASS_WEIGHT[int(round(np.clip(v[6], 0, 2)))],
        "criterion": DECODE_CRITERION[int(round(np.clip(v[7], 0, 2)))],
    }


# FUNCIONES DE NIVELACION DE CARGAS PARA MULTIPROCESSING
# Esta es la unica forma de paralelismo permitida en todo el programa, no se usa joblib ni n_jobs.

def nivelacion_cargas(n_p, indices):
    # Reparte una lista de indices entre n_p procesos de la manera mas equitativa posible
    n_p = min(n_p, len(indices))
    if n_p <= 0:
        return [indices]
    s = len(indices) % n_p
    n_d = indices[:s]
    t = (len(indices) - s) // n_p
    out = []
    temp = []
    for i in indices[s:]:
        temp.append(i)
        if len(temp) == t:
            out.append(temp)
            temp = []
    for i in range(len(n_d)):
        out[i].append(n_d[i])
    return out


# EVALUACION DE UN INDIVIDUO, USADA POR AMBOS MODOS DE PARALELIZACION
# La funcion de fitness combina F1 y AP en validacion, exactamente igual en secuencial y paralelo.

def evaluar_individuo(v, X_tr_arr, y_tr_arr, X_val_arr, y_val_arr, n_estimators_workers=1):
    params = decodificar(v)
    if n_estimators_workers > 1:
        # Modo de paralelizacion RF: el bosque se entrena repartiendo los arboles entre procesos
        modelo = entrenar_rf_paralelo(params, X_tr_arr, y_tr_arr, n_estimators_workers)
    else:
        # Modo secuencial puro o modo de paralelizacion DE, el bosque se entrena de forma normal
        modelo = RandomForestClassifier(random_state=42, **params)
        modelo.fit(X_tr_arr, y_tr_arr)
    y_pred_val = modelo.predict(X_val_arr)
    y_prob_val = modelo.predict_proba(X_val_arr)[:, 1]
    f1 = f1_score(y_val_arr, y_pred_val, pos_label=1, zero_division=0)
    apr = average_precision_score(y_val_arr, y_prob_val)
    return -(0.7 * f1 + 0.3 * apr)


# ENTRENAMIENTO DEL BOSQUE ALEATORIO PARALELIZADO MANUALMENTE CON MULTIPROCESSING
# Aqui se reparten los arboles del bosque entre varios procesos usando la nivelacion de cargas,
# se entrena cada subconjunto de arboles por separado y luego se combinan en un solo modelo.

def _entrenar_subbosque(args):
    n_arboles, params_base, X_tr_arr, y_tr_arr, semilla_sub = args
    params = dict(params_base)
    params["n_estimators"] = n_arboles
    modelo = RandomForestClassifier(random_state=semilla_sub, **params)
    modelo.fit(X_tr_arr, y_tr_arr)
    return modelo.estimators_, modelo.classes_, modelo.n_outputs_, modelo.n_features_in_


def entrenar_rf_paralelo(params, X_tr_arr, y_tr_arr, n_workers):
    n_total = params["n_estimators"]
    params_sin_n = {k: v for k, v in params.items() if k != "n_estimators"}
    indices = list(range(n_total))
    grupos = nivelacion_cargas(n_workers, indices)
    tareas = []
    for i, grupo in enumerate(grupos):
        tareas.append((len(grupo), params_sin_n, X_tr_arr, y_tr_arr, 1000 + i))
    n_proc = min(n_workers, len(tareas))
    with mp.Pool(processes=n_proc) as pool:
        resultados = pool.map(_entrenar_subbosque, tareas)

    arboles_totales = []
    for arboles, clases, n_out, n_feat in resultados:
        arboles_totales.extend(arboles)
    # Se construye un modelo final usando uno de los subbosques como base y se reemplazan sus arboles
    modelo_final = RandomForestClassifier(random_state=42, **params)
    modelo_final.n_estimators = len(arboles_totales)
    modelo_final.estimators_ = arboles_totales
    modelo_final.classes_ = resultados[0][1]
    modelo_final.n_classes_ = len(resultados[0][1])
    modelo_final.n_outputs_ = resultados[0][2]
    modelo_final.n_features_in_ = resultados[0][3]
    return modelo_final


# WORKER PARA LA PARALELIZACION EN LA EVOLUCION DIFERENCIAL
# Cada proceso recibe un bloque de individuos de la poblacion y los evalua de manera secuencial,
# el reparto de individuos entre procesos se hace con la misma funcion de nivelacion de cargas.

def _worker_chunk_de(args_chunk):
    resultados = []
    for idx, v, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr in args_chunk:
        t0 = time.time()
        fit = evaluar_individuo(v, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr, n_estimators_workers=1)
        resultados.append((idx, fit, time.time() - t0))
    return resultados


def evaluar_poblacion_de_paralelo(vectores, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr, n_workers):
    n = len(vectores)
    indices = list(range(n))
    chunks_idx = nivelacion_cargas(n_workers, indices)
    args = [[(idx, vectores[idx], X_tr_arr, y_tr_arr, X_v_arr, y_v_arr) for idx in chunk] for chunk in chunks_idx]
    n_proc = min(n_workers, len(args))
    with mp.Pool(processes=n_proc) as pool:
        resultados_chunks = pool.map(_worker_chunk_de, args)
    aptitud = [None] * n
    tiempos = []
    for chunk in resultados_chunks:
        for idx, fit, t_eval in chunk:
            aptitud[idx] = fit
            tiempos.append(t_eval)
    return np.array(aptitud), tiempos


def evaluar_poblacion_secuencial(vectores, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr, n_workers_rf):
    # Recorre la poblacion uno por uno, igual que el paralelo pero sin repartir trabajo entre procesos.
    # Si el modo activo es RF, cada evaluacion individual usa el entrenamiento de bosque paralelizado,
    # para que la unica diferencia entre secuencial y paralelo sea el uso o no de multiprocessing.
    aptitud = np.zeros(len(vectores))
    tiempos = []
    for i, v in enumerate(vectores):
        t0 = time.time()
        aptitud[i] = evaluar_individuo(v, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr,
                                        n_estimators_workers=n_workers_rf)
        tiempos.append(time.time() - t0)
    return aptitud, tiempos


# CICLO PRINCIPAL DE LA EVOLUCION DIFERENCIAL
# La funcion sirve tanto para la version secuencial como para la paralela, el comportamiento
# depende del parametro paralelo y del modo de paralelizacion seleccionado globalmente.

def evolucion_diferencial(NP, F, CR, max_gen, paciencia, tol, semilla,
                           X_train, y_train, X_val, y_val,
                           paralelo, modo, n_workers):
    rng = np.random.default_rng(semilla)
    poblacion = rng.uniform(LIM_INF, LIM_SUP, size=(NP, N_DIMS))
    X_tr_arr = X_train.values
    y_tr_arr = y_train.values
    X_v_arr = X_val.values
    y_v_arr = y_val.values

    # En modo RF, el numero de workers para entrenar cada bosque es n_workers, tanto en secuencial
    # como en paralelo, ya que en secuencial cada bosque individual se entrena tambien con ese metodo.
    n_workers_rf = n_workers if modo == "RF" else 1

    tiempos = []
    inicio_total = time.time()

    if paralelo and modo == "DE":
        aptitud, t_init = evaluar_poblacion_de_paralelo(poblacion, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr, n_workers)
    else:
        aptitud, t_init = evaluar_poblacion_secuencial(poblacion, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr, n_workers_rf)
    tiempos.extend(t_init)

    mejor_idx = np.argmin(aptitud)
    mejor_apt = aptitud[mejor_idx]
    mejor_vec = poblacion[mejor_idx].copy()
    hist_mejor = [-mejor_apt]
    hist_prom = [-np.mean(aptitud)]
    hist_tiempos = [time.time() - inicio_total]
    generacion_parada = 0

    for gen in range(1, max_gen + 1):
        trials = []
        for i in range(NP):
            candidatos = [j for j in range(NP) if j != i]
            a, b, c = rng.choice(candidatos, size=3, replace=False)
            donante = poblacion[a] + F * (poblacion[b] - poblacion[c])
            mask = rng.random(N_DIMS) < CR
            mask[rng.integers(0, N_DIMS)] = True
            trial = np.clip(np.where(mask, donante, poblacion[i]), LIM_INF, LIM_SUP)
            trials.append(trial)

        if paralelo and modo == "DE":
            apt_trials, t_gen = evaluar_poblacion_de_paralelo(trials, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr, n_workers)
        else:
            apt_trials, t_gen = evaluar_poblacion_secuencial(trials, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr, n_workers_rf)
        tiempos.extend(t_gen)

        for i in range(NP):
            if apt_trials[i] <= aptitud[i]:
                poblacion[i] = trials[i]
                aptitud[i] = apt_trials[i]
                if apt_trials[i] < mejor_apt:
                    mejor_apt = apt_trials[i]
                    mejor_vec = trials[i].copy()

        hist_mejor.append(-mejor_apt)
        hist_prom.append(-np.mean(aptitud))
        hist_tiempos.append(time.time() - inicio_total)
        generacion_parada = gen

        # Criterio de parada temprana, si la mejora en la ventana de paciencia es menor a la tolerancia
        if gen >= paciencia:
            ventana = hist_mejor[-paciencia:]
            ref = hist_mejor[-paciencia - 1]
            if max(ventana) - ref < tol:
                break

    tiempo_total = time.time() - inicio_total
    return {
        "mejor_vec": mejor_vec.tolist(),
        "hist_mejor": hist_mejor,
        "hist_prom": hist_prom,
        "hist_tiempos": hist_tiempos,
        "tiempos_evaluacion": tiempos,
        "tiempo_total": tiempo_total,
        "generacion_parada": generacion_parada,
    }


# EVALUACION FINAL DEL MEJOR MODELO ENCONTRADO SOBRE EL CONJUNTO DE PRUEBA

def evaluar_modelo_final(mejor_vec, nombre, X_train_val, y_train_val, X_test, y_test, carpeta):
    params = decodificar(mejor_vec)
    modelo = RandomForestClassifier(random_state=42, **params)
    modelo.fit(X_train_val, y_train_val)

    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Clase 0", "Clase 1"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Matriz de confusion - {nombre}")
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, f"matriz_confusion_{nombre}.png"), dpi=150)
    plt.close(fig)

    return {"f1": float(f1), "recall": float(recall), "auc": float(auc), "params": params}


# FUNCION PRINCIPAL DEL PROGRAMA
# Carga los datos, ejecuta la version secuencial UNA VEZ como base de comparacion, y ejecuta
# automaticamente las DOS versiones paralelas, una con paralelizacion en la evolucion diferencial
# y otra con paralelizacion en el entrenamiento del bosque aleatorio. El switch entre ambos modos
# de paralelizacion es automatico, no requiere configuracion manual. Al final guarda todos los
# resultados en json y csv, y genera las graficas comparativas para los tres casos.

def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    X_train = pd.read_csv(RUTA_X_TRAIN)
    y_train = pd.read_csv(RUTA_Y_TRAIN).squeeze()
    X_val = pd.read_csv(RUTA_X_VAL)
    y_val = pd.read_csv(RUTA_Y_VAL).squeeze()
    X_test = pd.read_csv(RUTA_X_TEST)
    y_test = pd.read_csv(RUTA_Y_TEST).squeeze()

    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    print("COMPARACION DE RENDIMIENTO: SECUENCIAL vs PARALELO (DE y RF)")
    print(f"NP = {NP}, generaciones maximas = {MAX_GEN}, workers = {N_WORKERS}")

    # LA VERSION SECUENCIAL SE EJECUTA UNA SOLA VEZ Y SIRVE DE BASE PARA LOS DOS MODOS PARALELOS
    print("\nEjecutando version secuencial (base de comparacion)...")
    t0 = time.time()
    res_seq = evolucion_diferencial(NP, F, CR, MAX_GEN, PACIENCIA, TOL, SEMILLA,
                                     X_train, y_train, X_val, y_val,
                                     paralelo=False, modo="DE", n_workers=N_WORKERS)
    tiempo_seq = time.time() - t0

    print("Ejecutando version paralela con paralelizacion en evolucion diferencial (DE)...")
    t0 = time.time()
    res_par_de = evolucion_diferencial(NP, F, CR, MAX_GEN, PACIENCIA, TOL, SEMILLA,
                                        X_train, y_train, X_val, y_val,
                                        paralelo=True, modo="DE", n_workers=N_WORKERS)
    tiempo_par_de = time.time() - t0

    print("Ejecutando version paralela con paralelizacion en bosques aleatorios (RF)...")
    t0 = time.time()
    res_par_rf = evolucion_diferencial(NP, F, CR, MAX_GEN, PACIENCIA, TOL, SEMILLA,
                                        X_train, y_train, X_val, y_val,
                                        paralelo=True, modo="RF", n_workers=N_WORKERS)
    tiempo_par_rf = time.time() - t0

    speedup_de = tiempo_seq / tiempo_par_de if tiempo_par_de > 0 else float("nan")
    eficiencia_de = speedup_de / N_WORKERS
    speedup_rf = tiempo_seq / tiempo_par_rf if tiempo_par_rf > 0 else float("nan")
    eficiencia_rf = speedup_rf / N_WORKERS

    print("\nRESULTADOS DE TIEMPOS")
    print(f"Tiempo secuencial:           {tiempo_seq:.2f} s")
    print(f"Tiempo paralelo (modo DE):   {tiempo_par_de:.2f} s")
    print(f"Tiempo paralelo (modo RF):   {tiempo_par_rf:.2f} s")
    print(f"Speedup modo DE:             {speedup_de:.2f}x (eficiencia {eficiencia_de*100:.1f}%)")
    print(f"Speedup modo RF:             {speedup_rf:.2f}x (eficiencia {eficiencia_rf*100:.1f}%)")

    # Evaluacion final de los tres modelos sobre el conjunto de prueba
    print("\nEvaluacion final en test...")
    final_seq = evaluar_modelo_final(res_seq["mejor_vec"], "secuencial",
                                      X_train_val.values, y_train_val.values,
                                      X_test.values, y_test.values, CARPETA_SALIDA)
    final_par_de = evaluar_modelo_final(res_par_de["mejor_vec"], "paralelo_DE",
                                         X_train_val.values, y_train_val.values,
                                         X_test.values, y_test.values, CARPETA_SALIDA)
    final_par_rf = evaluar_modelo_final(res_par_rf["mejor_vec"], "paralelo_RF",
                                         X_train_val.values, y_train_val.values,
                                         X_test.values, y_test.values, CARPETA_SALIDA)

    # GUARDADO DE RESULTADOS EN JSON, ESTE ARCHIVO ES LA FUENTE PRINCIPAL PARA LAS GRAFICAS POSTERIORES
    resultados_json = {
        "configuracion": {"NP": NP, "F": F, "CR": CR, "MAX_GEN": MAX_GEN,
                          "PACIENCIA": PACIENCIA, "TOL": TOL, "SEMILLA": SEMILLA,
                          "N_WORKERS": N_WORKERS},
        "tiempos": {
            "secuencial": tiempo_seq,
            "paralelo_DE": tiempo_par_de,
            "paralelo_RF": tiempo_par_rf,
            "speedup_DE": speedup_de,
            "eficiencia_DE": eficiencia_de,
            "speedup_RF": speedup_rf,
            "eficiencia_RF": eficiencia_rf,
        },
        "secuencial": res_seq,
        "paralelo_DE": res_par_de,
        "paralelo_RF": res_par_rf,
        "evaluacion_final": {
            "secuencial": final_seq,
            "paralelo_DE": final_par_de,
            "paralelo_RF": final_par_rf,
        },
    }
    with open(os.path.join(CARPETA_SALIDA, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(resultados_json, f, indent=2, ensure_ascii=False)

    # GUARDADO DE RESULTADOS EN CSV, PARA FACILITAR EL ANALISIS TABULAR
    filas = []
    for etiqueta, res in [("secuencial", res_seq), ("paralelo_DE", res_par_de), ("paralelo_RF", res_par_rf)]:
        for i, (mejor, prom, tacum) in enumerate(zip(res["hist_mejor"], res["hist_prom"], res["hist_tiempos"])):
            filas.append({"version": etiqueta, "generacion": i, "mejor_fitness": mejor,
                           "fitness_promedio": prom, "tiempo_acumulado": tacum})
    pd.DataFrame(filas).to_csv(os.path.join(CARPETA_SALIDA, "historial_convergencia.csv"), index=False)

    # GENERACION DE GRAFICAS COMPARATIVAS, CON ENFASIS EN EL SPEEDUP Y LOS RESULTADOS PRINCIPALES
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparacion Secuencial vs Paralelo (DE y RF)", fontsize=14)

    axes[0, 0].plot(range(len(res_seq["hist_mejor"])), res_seq["hist_mejor"], "o-", label="Secuencial")
    axes[0, 0].plot(range(len(res_par_de["hist_mejor"])), res_par_de["hist_mejor"], "s-", label="Paralelo DE")
    axes[0, 0].plot(range(len(res_par_rf["hist_mejor"])), res_par_rf["hist_mejor"], "^-", label="Paralelo RF")
    axes[0, 0].set_xlabel("Generacion")
    axes[0, 0].set_ylabel("Mejor fitness")
    axes[0, 0].set_title("Convergencia por generacion")
    axes[0, 0].legend()

    axes[0, 1].plot(res_seq["hist_tiempos"], res_seq["hist_mejor"], "o-", label="Secuencial")
    axes[0, 1].plot(res_par_de["hist_tiempos"], res_par_de["hist_mejor"], "s-", label="Paralelo DE")
    axes[0, 1].plot(res_par_rf["hist_tiempos"], res_par_rf["hist_mejor"], "^-", label="Paralelo RF")
    axes[0, 1].set_xlabel("Tiempo acumulado (s)")
    axes[0, 1].set_ylabel("Mejor fitness")
    axes[0, 1].set_title("Fitness vs tiempo")
    axes[0, 1].legend()

    etiquetas_tiempo = ["Secuencial", "Paralelo DE", "Paralelo RF"]
    valores_tiempo = [tiempo_seq, tiempo_par_de, tiempo_par_rf]
    axes[1, 0].bar(etiquetas_tiempo, valores_tiempo, color=["#2E86AB", "#A23B72", "#F18F01"])
    axes[1, 0].set_ylabel("Tiempo total (s)")
    axes[1, 0].set_title("Tiempo total de ejecucion")

    etiquetas_speedup = ["Speedup DE", "Eficiencia DE", "Speedup RF", "Eficiencia RF"]
    valores_speedup = [speedup_de, eficiencia_de, speedup_rf, eficiencia_rf]
    axes[1, 1].bar(etiquetas_speedup, valores_speedup, color=["#2E86AB", "#A9D6E5", "#F18F01", "#FBC490"])
    axes[1, 1].set_title("Speedup y eficiencia por modo")
    for i, v in enumerate(valores_speedup):
        axes[1, 1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDA, "comparacion_speedup.png"), dpi=150)
    plt.close(fig)

    print(f"\nMetricas finales secuencial:  F1={final_seq['f1']:.4f}, Recall={final_seq['recall']:.4f}, AUC={final_seq['auc']:.4f}")
    print(f"Metricas finales paralelo DE: F1={final_par_de['f1']:.4f}, Recall={final_par_de['recall']:.4f}, AUC={final_par_de['auc']:.4f}")
    print(f"Metricas finales paralelo RF: F1={final_par_rf['f1']:.4f}, Recall={final_par_rf['recall']:.4f}, AUC={final_par_rf['auc']:.4f}")
    print(f"\nResultados guardados en la carpeta '{CARPETA_SALIDA}'")


if __name__ == "__main__":
    main()