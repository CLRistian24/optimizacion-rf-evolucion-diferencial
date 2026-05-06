import time
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score

#  CONFIGURACIÓN COMÚN 
RUTA_X_TRAIN = "Archivos/Division/X_train.csv"
RUTA_Y_TRAIN = "Archivos/Division/y_train.csv"
RUTA_X_VAL   = "Archivos/Division/X_val.csv"
RUTA_Y_VAL   = "Archivos/Division/y_val.csv"
RUTA_X_TEST  = "Archivos/Division/X_test.csv"
RUTA_Y_TEST  = "Archivos/Division/y_test.csv"

X_train = pd.read_csv(RUTA_X_TRAIN)
y_train = pd.read_csv(RUTA_Y_TRAIN).squeeze()
X_val   = pd.read_csv(RUTA_X_VAL)
y_val   = pd.read_csv(RUTA_Y_VAL).squeeze()
X_test  = pd.read_csv(RUTA_X_TEST)
y_test  = pd.read_csv(RUTA_Y_TEST).squeeze()

X_train_val = pd.concat([X_train, X_val], ignore_index=True)
y_train_val = pd.concat([y_train, y_val], ignore_index=True)

# Espacio de búsqueda (igual para ambos)
ESPACIO = {
    "n_estimators":      (100.0, 500.0),
    "max_depth":         (3.0,   50.0),
    "min_samples_split": (2.0,   50.0),
    "min_samples_leaf":  (1.0,   20.0),
    "max_features":      (0.1,   1.0),
    "bootstrap":         (0.0,   1.0),
    "class_weight":      (0.0,   2.0),
    "criterion":         (0.0,   2.0),
}
NOMBRES = list(ESPACIO.keys())
N_DIMS  = len(NOMBRES)
lim_inf = np.array([ESPACIO[k][0] for k in NOMBRES])
lim_sup = np.array([ESPACIO[k][1] for k in NOMBRES])

DECODE_BOOTSTRAP    = [False, True]
DECODE_CLASS_WEIGHT = [None, "balanced", "balanced_subsample"]
DECODE_CRITERION    = ["gini", "entropy", "log_loss"]

def decodificar(v):
    return {
        "n_estimators":      int(round(v[0])),
        "max_depth":         int(round(v[1])),
        "min_samples_split": int(round(v[2])),
        "min_samples_leaf":  int(round(v[3])),
        "max_features":      float(np.clip(v[4], 0.1, 1.0)),
        "bootstrap":         DECODE_BOOTSTRAP[int(round(np.clip(v[5], 0, 1)))],
        "class_weight":      DECODE_CLASS_WEIGHT[int(round(np.clip(v[6], 0, 2)))],
        "criterion":         DECODE_CRITERION[int(round(np.clip(v[7], 0, 2)))],
    }

def evaluar_individuo(v, X_tr_arr, y_tr_arr, X_val_arr, y_val_arr):
    """Evalúa un individuo sin validación cruzada (entrena en train, valida en val)"""
    params = decodificar(v)
    modelo = RandomForestClassifier(random_state=42, **params)
    modelo.fit(X_tr_arr, y_tr_arr)
    y_pred_val = modelo.predict(X_val_arr)
    y_prob_val = modelo.predict_proba(X_val_arr)[:, 1]
    f1 = f1_score(y_val_arr, y_pred_val, pos_label=1, zero_division=0)
    apr = average_precision_score(y_val_arr, y_prob_val)
    return -(0.7 * f1 + 0.3 * apr)

# VERSIÓN SECUENCIAL
def de_sequential(NP, F, CR, max_gen, paciencia, tol, semilla):
    rng = np.random.default_rng(semilla)
    poblacion = rng.uniform(lim_inf, lim_sup, size=(NP, N_DIMS))
    X_tr_arr = X_train.values
    y_tr_arr = y_train.values
    X_v_arr  = X_val.values
    y_v_arr  = y_val.values

    tiempos = []
    inicio_total = time.time()
    aptitud = np.zeros(NP)
    for i, ind in enumerate(poblacion):
        t0 = time.time()
        aptitud[i] = evaluar_individuo(ind, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr)
        tiempos.append(time.time() - t0)

    mejor_idx = np.argmin(aptitud)
    mejor_apt = aptitud[mejor_idx]
    mejor_vec = poblacion[mejor_idx].copy()
    hist_mejor = [-mejor_apt]
    hist_prom = [-np.mean(aptitud)]
    hist_tiempos = [time.time() - inicio_total]
    generacion_parada = 0

    for gen in range(1, max_gen + 1):
        for i in range(NP):
            candidatos = [j for j in range(NP) if j != i]
            a, b, c = rng.choice(candidatos, size=3, replace=False)
            donante = poblacion[a] + F * (poblacion[b] - poblacion[c])
            mask = rng.random(N_DIMS) < CR
            mask[rng.integers(0, N_DIMS)] = True
            trial = np.clip(np.where(mask, donante, poblacion[i]), lim_inf, lim_sup)
            t0 = time.time()
            f_trial = evaluar_individuo(trial, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr)
            tiempos.append(time.time() - t0)
            if f_trial <= aptitud[i]:
                poblacion[i] = trial
                aptitud[i] = f_trial
                if f_trial < mejor_apt:
                    mejor_apt = f_trial
                    mejor_vec = trial.copy()
        hist_mejor.append(-mejor_apt)
        hist_prom.append(-np.mean(aptitud))
        hist_tiempos.append(time.time() - inicio_total)
        generacion_parada = gen
        # Parada temprana simple
        if gen >= paciencia:
            ventana = hist_mejor[-paciencia:]
            ref = hist_mejor[-paciencia-1]
            if max(ventana) - ref < tol:
                break
    tiempo_total = time.time() - inicio_total
    return mejor_vec, hist_mejor, hist_prom, hist_tiempos, tiempos, tiempo_total, generacion_parada

#  VERSIÓN PARALELA 
# Funciones auxiliares para paralelismo
def nivelacion_cargas(n_p, indices):
    n_p = min(n_p, len(indices))
    s = len(indices) % n_p
    n_D = indices[:s]
    t = (len(indices) - s) // n_p
    out = []
    temp = []
    for i in indices[s:]:
        temp.append(i)
        if len(temp) == t:
            out.append(temp)
            temp = []
    for i in range(len(n_D)):
        out[i].append(n_D[i])
    return out

def _worker_chunk(args_chunk):
    resultados = []
    for idx, v, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr in args_chunk:
        t0 = time.time()
        fit = evaluar_individuo(v, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr)
        resultados.append((idx, fit, time.time() - t0))
    return resultados

def evaluar_poblacion_paralelo(vectores, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr, n_workers):
    n = len(vectores)
    indices = list(range(n))
    n_proc = min(n_workers, n)
    chunks_idx = nivelacion_cargas(n_proc, indices)
    args = [[(idx, vectores[idx], X_tr_arr, y_tr_arr, X_v_arr, y_v_arr) for idx in chunk] for chunk in chunks_idx]
    with mp.Pool(processes=n_proc) as pool:
        resultados_chunks = pool.map(_worker_chunk, args)
    aptitud = [None] * n
    tiempos = []
    for chunk in resultados_chunks:
        for idx, fit, t_eval in chunk:
            aptitud[idx] = fit
            tiempos.append(t_eval)
    return np.array(aptitud), tiempos

def de_parallel(NP, F, CR, max_gen, paciencia, tol, semilla, n_workers):
    rng = np.random.default_rng(semilla)
    poblacion = rng.uniform(lim_inf, lim_sup, size=(NP, N_DIMS))
    X_tr_arr = X_train.values
    y_tr_arr = y_train.values
    X_v_arr  = X_val.values
    y_v_arr  = y_val.values

    tiempos = []
    inicio_total = time.time()
    aptitud, t_init = evaluar_poblacion_paralelo(poblacion, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr, n_workers)
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
            trial = np.clip(np.where(mask, donante, poblacion[i]), lim_inf, lim_sup)
            trials.append(trial)
        apt_trials, t_gen = evaluar_poblacion_paralelo(trials, X_tr_arr, y_tr_arr, X_v_arr, y_v_arr, n_workers)
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
        if gen >= paciencia:
            ventana = hist_mejor[-paciencia:]
            ref = hist_mejor[-paciencia-1]
            if max(ventana) - ref < tol:
                break
    tiempo_total = time.time() - inicio_total
    return mejor_vec, hist_mejor, hist_prom, hist_tiempos, tiempos, tiempo_total, generacion_parada

#  EJECUCIÓN DE COMPARACIÓN 
print("COMPARACIÓN DE RENDIMIENTO: SECUENCIAL vs PARALELO")


# Parámetros DE
NP = 4
F = 0.8
CR = 0.9
MAX_GEN = 3
PACIENCIA = 2
TOL = 1e-4
SEMILLA = 42
N_WORKERS = mp.cpu_count() - 1  # todos menos uno

print(f"\nConfiguración:")
print(f"  NP = {NP}, Generaciones máx = {MAX_GEN}")
print(f"  Workers (paralelo) = {N_WORKERS}")

print("\nEjecutando versión secuencial...")
seq_start = time.time()
mejor_vec_seq, hist_mejor_seq, hist_prom_seq, hist_t_seq, tiempos_seq, tiempo_total_seq, gen_parada_seq = de_sequential(
    NP, F, CR, MAX_GEN, PACIENCIA, TOL, SEMILLA)
tiempo_seq = time.time() - seq_start

print("\nEjecutando versión paralela...")
par_start = time.time()
mejor_vec_par, hist_mejor_par, hist_prom_par, hist_t_par, tiempos_par, tiempo_total_par, gen_parada_par = de_parallel(
    NP, F, CR, MAX_GEN, PACIENCIA, TOL, SEMILLA, N_WORKERS)
tiempo_par = time.time() - par_start

# Cálculo de speedup y eficiencia
speedup = tiempo_seq / tiempo_par
eficiencia = speedup / N_WORKERS

print("RESULTADOS DE TIEMPOS")
print(f"Tiempo secuencial:          {tiempo_seq:.2f} s")
print(f"Tiempo paralelo:            {tiempo_par:.2f} s")
print(f"Speedup (S = T_seq/T_par):  {speedup:.2f}x")
print(f"Nivel de paralelismo usado: {N_WORKERS} workers")
print(f"Eficiencia (S / workers):   {eficiencia:.2f} ({eficiencia*100:.1f}%)")

# Comparar calidad de la solución (fitness final)
mejor_fitness_seq = max(hist_mejor_seq)  # porque guardamos negativo
mejor_fitness_par = max(hist_mejor_par)
print("\n" + "="*60)
print("CALIDAD DE LA SOLUCIÓN (Fitness en validación)")
print("="*60)
print(f"Mejor fitness secuencial: {mejor_fitness_seq:.6f}")
print(f"Mejor fitness paralelo:   {mejor_fitness_par:.6f}")
print(f"Diferencia:               {abs(mejor_fitness_seq - mejor_fitness_par):.6f}")

# Gráficas comparativas
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Comparación Secuencial vs Paralelo", fontsize=14)

# 1. Convergencia
gens_seq = range(len(hist_mejor_seq))
gens_par = range(len(hist_mejor_par))
axes[0,0].plot(gens_seq, hist_mejor_seq, 'o-', label='Secuencial', color='C0')
axes[0,0].plot(gens_par, hist_mejor_par, 's-', label='Paralelo', color='C1')
axes[0,0].set_xlabel("Generación")
axes[0,0].set_ylabel("Mejor Fitness")
axes[0,0].set_title("Curva de Convergencia")
axes[0,0].legend()
axes[0,0].grid(True)

# 2. Tiempo acumulado 
axes[0,1].plot(hist_t_seq, hist_mejor_seq, 'o-', label='Secuencial', color='C0')
axes[0,1].plot(hist_t_par, hist_mejor_par, 's-', label='Paralelo', color='C1')
axes[0,1].set_xlabel("Tiempo acumulado (s)")
axes[0,1].set_ylabel("Mejor Fitness")
axes[0,1].set_title("Fitness vs Tiempo")
axes[0,1].legend()
axes[0,1].grid(True)

# 3. Distribución de tiempos de evaluación (boxplot)
data_eval = [tiempos_seq, tiempos_par]
axes[1,0].boxplot(data_eval, labels=['Secuencial', 'Paralelo'], patch_artist=True,
                  boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red'))
axes[1,0].set_ylabel("Tiempo por evaluación (s)")
axes[1,0].set_title("Distribución de tiempos de evaluación")
axes[1,0].grid(axis='y')

# 4. Speedup y eficiencia (barra)
metricas = ['Speedup', 'Eficiencia']
valores = [speedup, eficiencia]
axes[1,1].bar(metricas, valores, color=['#2E86AB', '#A23B72'])
axes[1,1].set_ylabel("Valor")
axes[1,1].set_title("Speedup y Eficiencia")
axes[1,1].grid(axis='y')
for i, v in enumerate(valores):
    axes[1,1].text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("comparacion_speedup.png", dpi=150)
plt.close()
print("\nGráfica guardada: comparacion_speedup.png")

# Guardar resultados detallados en archivo
with open("resultados_comparacion.txt", "w", encoding="utf-8") as f:
    f.write("COMPARACIÓN SECUENCIAL vs PARALELO\n")
    f.write("="*60 + "\n")
    f.write(f"Parámetros DE: NP={NP}, max_gen={MAX_GEN}, workers={N_WORKERS}\n\n")
    f.write("TIEMPOS\n")
    f.write(f"  Secuencial: {tiempo_seq:.2f} s\n")
    f.write(f"  Paralelo:   {tiempo_par:.2f} s\n")
    f.write(f"  Speedup:    {speedup:.2f}x\n")
    f.write(f"  Eficiencia: {eficiencia:.2f} ({eficiencia*100:.1f}%)\n\n")
    f.write("CALIDAD\n")
    f.write(f"  Fitness final secuencial: {mejor_fitness_seq:.6f}\n")
    f.write(f"  Fitness final paralelo:   {mejor_fitness_par:.6f}\n\n")
    f.write("ESTADÍSTICAS DE EVALUACIONES\n")
    f.write(f"  Secuencial - media: {np.mean(tiempos_seq):.3f}s, std: {np.std(tiempos_seq):.3f}s\n")
    f.write(f"  Paralelo   - media: {np.mean(tiempos_par):.3f}s, std: {np.std(tiempos_par):.3f}s\n")

print("Resultados guardados en 'resultados_comparacion.txt'")