import time
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    f1_score,
    recall_score,
    ConfusionMatrixDisplay,
)

# Rutas (iguales)
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

X_trainval = pd.concat([X_train, X_val], ignore_index=True)
y_trainval = pd.concat([y_train, y_val], ignore_index=True)

print("\nTamaños de conjuntos:")
print(f"  Entrenamiento: {X_train.shape}, fraudes={y_train.sum()}")
print(f"  Validación:    {X_val.shape},  fraudes={y_val.sum()}")
print(f"  Prueba:        {X_test.shape},  fraudes={y_test.sum()}")

# Espacio de búsqueda (8 parámetros)
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


def evaluar_individuo(v, X_tr, y_tr, X_vald, y_vald):
    """Evalúa un individuo entrenando en train y validando en val (sin CV)"""
    params = decodificar(v)
    modelo = RandomForestClassifier(random_state=42, **params)
    modelo.fit(X_tr, y_tr)
    y_prob_val = modelo.predict_proba(X_vald)[:, 1]
    y_pred_val = modelo.predict(X_vald)
    f1 = f1_score(y_vald, y_pred_val, pos_label=1, zero_division=0)
    apr = average_precision_score(y_vald, y_prob_val)
    fitness = 0.7 * f1 + 0.3 * apr
    return -fitness  # minimizamos


# Parámetros DE
NP               = 8        # Tamaño de población
F                = 0.8
CR               = 0.9
MAX_GENERACIONES = 10        # Aumentado para ver convergencia
PACIENCIA        = 3
TOL_MEJORA       = 1e-4
SEMILLA          = 42

rng = np.random.default_rng(SEMILLA)

lim_inf = np.array([ESPACIO[k][0] for k in NOMBRES])
lim_sup = np.array([ESPACIO[k][1] for k in NOMBRES])

poblacion = rng.uniform(lim_inf, lim_sup, size=(NP, N_DIMS))

print(f"\nEvolución Diferencial Secuencial: NP={NP}, F={F}, CR={CR}, max_gen={MAX_GENERACIONES}, paciencia={PACIENCIA}, tol={TOL_MEJORA}")

# Preparamos arrays para pasar a la función (evitando conversiones repetidas)
X_train_arr = X_train.values
y_train_arr = y_train.values
X_val_arr   = X_val.values
y_val_arr   = y_val.values

print("Evaluando población inicial...")
t_inicio_ed = time.time()
tiempos_evaluacion = []

aptitud = np.zeros(NP)
for i, ind in enumerate(poblacion):
    t0 = time.time()
    aptitud[i] = evaluar_individuo(ind, X_train_arr, y_train_arr, X_val_arr, y_val_arr)
    tiempos_evaluacion.append(time.time() - t0)

mejor_idx     = int(np.argmin(aptitud))
mejor_aptitud = aptitud[mejor_idx]
mejor_vector  = poblacion[mejor_idx].copy()

historial_mejores   = [-mejor_aptitud]
historial_promedio  = [-np.mean(aptitud)]
historial_tiempos   = [time.time() - t_inicio_ed]
generacion_parada   = 0
razon_parada        = f"máximo de generaciones ({MAX_GENERACIONES}) alcanzado"

print(f"  Gen   0 | mejor fitness: {historial_mejores[-1]:.4f} | promedio: {historial_promedio[-1]:.4f}")

for gen in range(1, MAX_GENERACIONES + 1):
    for i in range(NP):
        # Selección de 3 individuos distintos
        candidatos = [j for j in range(NP) if j != i]
        if len(candidatos) >= 3:
            a, b, c = rng.choice(candidatos, size=3, replace=False)
        else:
            a, b, c = rng.choice(candidatos, size=3, replace=True)

        donante = poblacion[a] + F * (poblacion[b] - poblacion[c])
        mascara = rng.random(N_DIMS) < CR
        mascara[rng.integers(0, N_DIMS)] = True
        trial = np.where(mascara, donante, poblacion[i])
        trial = np.clip(trial, lim_inf, lim_sup)

        t0 = time.time()
        f_trial = evaluar_individuo(trial, X_train_arr, y_train_arr, X_val_arr, y_val_arr)
        tiempos_evaluacion.append(time.time() - t0)

        if f_trial <= aptitud[i]:
            poblacion[i] = trial
            aptitud[i]   = f_trial
            if f_trial < mejor_aptitud:
                mejor_aptitud = f_trial
                mejor_vector  = trial.copy()

    historial_mejores.append(-mejor_aptitud)
    historial_promedio.append(-np.mean(aptitud))
    historial_tiempos.append(time.time() - t_inicio_ed)
    generacion_parada = gen

    print(f"  Gen {gen:3d} | mejor fitness: {historial_mejores[-1]:.4f} | promedio: {historial_promedio[-1]:.4f}")

    # Criterio de parada temprana
    if gen >= PACIENCIA:
        ventana    = historial_mejores[-PACIENCIA:]
        referencia = historial_mejores[-PACIENCIA - 1]
        mejora     = max(ventana) - referencia
        if mejora < TOL_MEJORA:
            razon_parada = f"parada temprana en gen {gen}: mejora={mejora:.6f} < {TOL_MEJORA}"
            print(f"\nParada temprana en generación {gen}: mejora={mejora:.6f} < {TOL_MEJORA}")
            break

tiempo_ed = time.time() - t_inicio_ed
mejores_params = decodificar(mejor_vector)

print("\nMejores hiperparámetros encontrados:")
for k, v in mejores_params.items():
    print(f"  {k}: {v}")
print(f"\nMejor fitness (validación directa): {historial_mejores[-1]:.4f}")
print(f"Tiempo total DE:                    {tiempo_ed:.2f} s  ({tiempo_ed/60:.2f} min)")

# Entrenamiento final con train+val
print("\nEntrenando modelo final...")
t_inicio_train = time.time()
modelo_final = RandomForestClassifier(random_state=42, **mejores_params)
modelo_final.fit(X_trainval, y_trainval)
tiempo_train = time.time() - t_inicio_train

# Evaluación en test
t_inicio_pred = time.time()
y_pred = modelo_final.predict(X_test)
y_prob = modelo_final.predict_proba(X_test)[:, 1]
tiempo_pred = time.time() - t_inicio_pred

f1_fraude  = f1_score(y_test, y_pred, pos_label=1)
pr_auc     = average_precision_score(y_test, y_prob)
recall_pos = recall_score(y_test, y_pred, pos_label=1)
cm         = confusion_matrix(y_test, y_pred)
reporte    = classification_report(y_test, y_pred, target_names=["Legítima", "Fraude"])

print("\nResultados en conjunto de prueba:")
print(reporte)
print("Matriz de confusión:")
print(cm)
print(f"\nF1-Score clase fraude:          {f1_fraude:.4f}")
print(f"AUC-PR:                         {pr_auc:.4f}")
print(f"Recall clase positiva (fraude): {recall_pos:.4f}")
print(f"Tiempo de entrenamiento final:  {tiempo_train:.4f} s")
print(f"Tiempo de inferencia (prueba):  {tiempo_pred:.4f} s")

importancias = pd.Series(
    modelo_final.feature_importances_,
    index=X_test.columns,
).sort_values(ascending=False)

print("\nTop 10 características más importantes:")
print(importancias.head(10).to_string())

# Estadísticas de tiempos de evaluación
n_eval_total = len(tiempos_evaluacion)
t_eval_media = np.mean(tiempos_evaluacion)
t_eval_std   = np.std(tiempos_evaluacion)
t_eval_min   = np.min(tiempos_evaluacion)
t_eval_max   = np.max(tiempos_evaluacion)
t_eval_total = np.sum(tiempos_evaluacion)

print("\nEstadísticas de evaluaciones:")
print(f"  Evaluaciones totales:   {n_eval_total}")
print(f"  Tiempo medio/eval:      {t_eval_media:.3f} s")
print(f"  Desviación estándar:    {t_eval_std:.3f} s")
print(f"  Tiempo mín/máx:         {t_eval_min:.3f} s / {t_eval_max:.3f} s")
print(f"  Tiempo evaluando total: {t_eval_total:.2f} s")

# ==================== GRÁFICAS ====================
COLOR_MEJOR = "#00C9A7"
COLOR_PROM  = "#F7B731"
COLOR_FONDO = "#0F1117"
COLOR_TEXTO = "#E8EAF0"
COLOR_CUAD  = "#1A1D27"
COLOR_GRID  = "#2A2D3A"

plt.rcParams.update({
    "figure.facecolor": COLOR_FONDO,
    "axes.facecolor":   COLOR_CUAD,
    "axes.edgecolor":   COLOR_GRID,
    "axes.labelcolor":  COLOR_TEXTO,
    "xtick.color":      COLOR_TEXTO,
    "ytick.color":      COLOR_TEXTO,
    "text.color":       COLOR_TEXTO,
    "grid.color":       COLOR_GRID,
    "grid.linewidth":   0.6,
    "font.family":      "monospace",
})

gens = list(range(len(historial_mejores)))

fig = plt.figure(figsize=(18, 14), facecolor=COLOR_FONDO)
fig.suptitle("Evolución Diferencial Secuencial — Resultados y Rendimiento",
             fontsize=16, color=COLOR_TEXTO, y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(gens, historial_mejores, color=COLOR_MEJOR, linewidth=2.2, label="Mejor fitness", zorder=3)
ax1.plot(gens, historial_promedio, color=COLOR_PROM, linewidth=1.5, linestyle="--", label="Promedio población", zorder=2)
ax1.fill_between(gens, historial_promedio, historial_mejores, alpha=0.12, color=COLOR_MEJOR)
ax1.axvline(generacion_parada, color="#FF6B6B", linewidth=1.2, linestyle=":", label=f"Parada (gen {generacion_parada})")
ax1.set_title("Curva de Convergencia", fontsize=12, pad=8)
ax1.set_xlabel("Generación")
ax1.set_ylabel("Fitness (0.7·F1 + 0.3·AUC-PR)")
ax1.legend(fontsize=9, facecolor=COLOR_CUAD, edgecolor=COLOR_GRID)
ax1.grid(True)

ax2 = fig.add_subplot(gs[0, 2])
delta = [historial_mejores[i] - historial_mejores[i-1] for i in range(1, len(historial_mejores))]
colores_delta = [COLOR_MEJOR if d >= 0 else "#FF6B6B" for d in delta]
ax2.bar(range(1, len(delta)+1), delta, color=colores_delta, width=0.8)
ax2.axhline(0, color=COLOR_TEXTO, linewidth=0.8)
ax2.set_title("Mejora por Generación", fontsize=12, pad=8)
ax2.set_xlabel("Generación")
ax2.set_ylabel("Δ Fitness")
ax2.grid(True, axis="y")

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(historial_tiempos, historial_mejores, color=COLOR_MEJOR, linewidth=2)
ax3.scatter(historial_tiempos[0],  historial_mejores[0],  color="#FFFFFF",   s=50, zorder=5)
ax3.scatter(historial_tiempos[-1], historial_mejores[-1], color=COLOR_PROM,  s=80, marker="*", zorder=5)
ax3.set_title("Fitness vs Tiempo acumulado", fontsize=11, pad=8)
ax3.set_xlabel("Tiempo (s)")
ax3.set_ylabel("Fitness")
ax3.grid(True)

ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(tiempos_evaluacion, bins=30, color=COLOR_MEJOR, edgecolor=COLOR_FONDO, alpha=0.85)
ax4.axvline(t_eval_media, color=COLOR_PROM, linewidth=1.8, linestyle="--",
            label=f"Media: {t_eval_media:.2f}s")
ax4.set_title("Distribución Tiempos de Evaluación", fontsize=11, pad=8)
ax4.set_xlabel("Tiempo por evaluación (s)")
ax4.set_ylabel("Frecuencia")
ax4.legend(fontsize=9, facecolor=COLOR_CUAD, edgecolor=COLOR_GRID)
ax4.grid(True, axis="y")

ax5 = fig.add_subplot(gs[1, 2])
eval_acum = np.cumsum(tiempos_evaluacion)
ax5.plot(range(1, n_eval_total+1), eval_acum, color=COLOR_PROM, linewidth=1.8)
ax5.set_title("Tiempo Acumulado de Evaluaciones", fontsize=11, pad=8)
ax5.set_xlabel("Número de evaluación")
ax5.set_ylabel("Tiempo acumulado (s)")
ax5.grid(True)

ax6 = fig.add_subplot(gs[2, 0])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legítima", "Fraude"])
disp.plot(ax=ax6, colorbar=False, cmap="YlGn")
ax6.set_title("Matriz de Confusión (Test)", fontsize=11, pad=8)
ax6.tick_params(colors=COLOR_TEXTO)
ax6.xaxis.label.set_color(COLOR_TEXTO)
ax6.yaxis.label.set_color(COLOR_TEXTO)

ax7 = fig.add_subplot(gs[2, 1])
metricas_nombres = ["F1-Fraude", "AUC-PR", "Recall\nFraude"]
metricas_valores = [f1_fraude, pr_auc, recall_pos]
barras = ax7.barh(metricas_nombres, metricas_valores,
                  color=[COLOR_MEJOR, COLOR_PROM, "#A78BFA"], height=0.45)
ax7.set_xlim(0, 1.15)
ax7.axvline(1.0, color=COLOR_TEXTO, linewidth=0.7, linestyle=":")
for bar, val in zip(barras, metricas_valores):
    ax7.text(val + 0.02, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", fontsize=10, color=COLOR_TEXTO)
ax7.set_title("Métricas en Conjunto de Prueba", fontsize=11, pad=8)
ax7.grid(True, axis="x")

ax8 = fig.add_subplot(gs[2, 2])
top10 = importancias.head(10)
ax8.barh(top10.index[::-1], top10.values[::-1], color=COLOR_MEJOR, alpha=0.85)
ax8.set_title("Top 10 Características", fontsize=11, pad=8)
ax8.set_xlabel("Importancia")
ax8.grid(True, axis="x")

plt.savefig("graficas_ed_secuencial.png", dpi=150, bbox_inches="tight", facecolor=COLOR_FONDO)
plt.close()
print("\nGráficas guardadas en 'graficas_ed_secuencial.png'")

# Guardar resultados en archivo de texto
with open("resultados_rf_ed_secuencial.txt", "w", encoding="utf-8") as f:
    f.write("=== RESULTADOS RF + EVOLUCIÓN DIFERENCIAL SECUENCIAL (SIN VALIDACIÓN CRUZADA) ===\n\n")
    f.write(f"DE: NP={NP}, F={F}, CR={CR}, max_gen={MAX_GENERACIONES}, paciencia={PACIENCIA}, tol={TOL_MEJORA}\n")
    f.write(f"Criterio de parada: {razon_parada}\n\n")
    f.write("Mejores hiperparámetros:\n")
    for k, v in mejores_params.items():
        f.write(f"  {k}: {v}\n")
    f.write(f"\nMejor fitness (validación directa): {historial_mejores[-1]:.4f}\n\n")
    f.write(" TIEMPOS \n")
    f.write(f"  Tiempo total DE:                 {tiempo_ed:.2f} s  ({tiempo_ed/60:.2f} min)\n")
    f.write(f"  Tiempo entrenamiento final:      {tiempo_train:.4f} s\n")
    f.write(f"  Tiempo inferencia (test):        {tiempo_pred:.6f} s\n\n")
    f.write(" ESTADÍSTICAS DE EVALUACIONES \n")
    f.write(f"  Total de evaluaciones:           {n_eval_total}\n")
    f.write(f"  Tiempo medio por evaluación:     {t_eval_media:.3f} s\n")
    f.write(f"  Desviación estándar:             {t_eval_std:.3f} s\n")
    f.write(f"  Tiempo mínimo:                   {t_eval_min:.3f} s\n")
    f.write(f"  Tiempo máximo:                   {t_eval_max:.3f} s\n")
    f.write(f"  Tiempo CPU total:                {t_eval_total:.2f} s\n\n")
    f.write(" EVOLUCIÓN DEL FITNESS \n")
    f.write(f"{'Gen':>5}  {'Mejor':>10}  {'Promedio':>10}  {'T pared (s)':>12}\n")
    for i, (m, p, t) in enumerate(zip(historial_mejores, historial_promedio, historial_tiempos)):
        f.write(f"  {i:3d}    {m:.6f}    {p:.6f}    {t:10.2f}\n")
    f.write("\n MÉTRICAS EN PRUEBA \n")
    f.write(f"  F1-score (fraude): {f1_fraude:.4f}\n")
    f.write(f"  AUC-PR:            {pr_auc:.4f}\n")
    f.write(f"  Recall (fraude):   {recall_pos:.4f}\n")
    f.write(f"\nMatriz de confusión:\n{cm}\n\n")
    f.write("Informe de clasificación:\n")
    f.write(reporte)
    f.write("\nTop 10 características:\n")
    f.write(importancias.head(10).to_string())

print("Resultados guardados en 'resultados_rf_ed_secuencial.txt'")