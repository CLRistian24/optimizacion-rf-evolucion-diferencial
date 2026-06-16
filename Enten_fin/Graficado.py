import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              f1_score, recall_score, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier

plt.style.use("seaborn-v0_8-darkgrid")

CARPETA_SALIDA = sys.argv[1] if len(sys.argv) > 1 else "Resultados_DE_RF_Checkpoint"

RUTA_X_TRAIN = "Archivos/Division/X_train.csv"
RUTA_Y_TRAIN = "Archivos/Division/y_train.csv"
RUTA_X_VAL   = "Archivos/Division/X_val.csv"
RUTA_Y_VAL   = "Archivos/Division/y_val.csv"
RUTA_X_TEST  = "Archivos/Division/X_test.csv"
RUTA_Y_TEST  = "Archivos/Division/y_test.csv"

DECODE_BOOTSTRAP    = [False, True]
DECODE_CLASS_WEIGHT = ["balanced", None, "balanced_subsample"]
DECODE_CRITERION    = ["entropy", "gini"]


def cargar_resultados():
    ruta = os.path.join(CARPETA_SALIDA, "resultados.json")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró '{ruta}'.")
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def cargar_datos():
    X_train = pd.read_csv(RUTA_X_TRAIN)
    y_train = pd.read_csv(RUTA_Y_TRAIN).squeeze()
    X_val   = pd.read_csv(RUTA_X_VAL)
    y_val   = pd.read_csv(RUTA_Y_VAL).squeeze()
    X_test  = pd.read_csv(RUTA_X_TEST)
    y_test  = pd.read_csv(RUTA_Y_TEST).squeeze()
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    return X_train_val.values, y_train_val.values, X_test.values, y_test.values


def reconstruir_modelo(params_dict, X_train_val, y_train_val):
    modelo = RandomForestClassifier(random_state=42, **params_dict)
    modelo.fit(X_train_val, y_train_val)
    return modelo


def guardar(fig, nombre):
    ruta = os.path.join(CARPETA_SALIDA, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {ruta}")


def decodificar(v):
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


def grafica_speedup_vs_workers(data, t_seq):
    workers, sp_de_list, ef_de_list, sp_rf_list, ef_rf_list = [], [], [], [], []
    for key, val in data["resultados_por_workers"].items():
        if not key.startswith("workers_"):
            continue
        w = val["n_workers"]
        workers.append(w)
        sp_de_list.append(val["paralelo_DE"]["speedup"])
        ef_de_list.append(val["paralelo_DE"]["eficiencia"])
        sp_rf_list.append(val["paralelo_RF"]["speedup"])
        ef_rf_list.append(val["paralelo_RF"]["eficiencia"])

    idx = np.argsort(workers)
    workers = np.array(workers)[idx]
    sp_de_list = np.array(sp_de_list)[idx]
    ef_de_list = np.array(ef_de_list)[idx]
    sp_rf_list = np.array(sp_rf_list)[idx]
    ef_rf_list = np.array(ef_rf_list)[idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(workers, sp_de_list, 'o-', color="#A23B72", label="Paralelo DE", markersize=8)
    ax1.plot(workers, sp_rf_list, 's-', color="#F18F01", label="Paralelo RF", markersize=8)
    ax1.plot(workers, workers, '--', color="gray", label="Speedup ideal")
    ax1.set_xlabel("Número de workers")
    ax1.set_ylabel("Speedup")
    ax1.set_title("Speedup vs Número de núcleos")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(workers, ef_de_list, 'o-', color="#A23B72", label="Paralelo DE", markersize=8)
    ax2.plot(workers, ef_rf_list, 's-', color="#F18F01", label="Paralelo RF", markersize=8)
    ax2.axhline(1.0, color="gray", linestyle="--", label="Eficiencia = 1")
    ax2.set_xlabel("Número de workers")
    ax2.set_ylabel("Eficiencia")
    ax2.set_title("Eficiencia vs Número de núcleos")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle("Rendimiento paralelo (DE y RF)", fontsize=14)
    plt.tight_layout()
    guardar(fig, "speedup_vs_workers.png")


def grafica_comparacion_general(data):
    res_seq    = data["secuencial"]
    res_par_de = data["paralelo_DE"]
    res_par_rf = data["paralelo_RF"]
    tiempos_d  = data["tiempos"]

    t_seq = tiempos_d["secuencial"]
    t_de  = tiempos_d["paralelo_DE"]
    t_rf  = tiempos_d["paralelo_RF"]
    sp_de = tiempos_d["speedup_DE"]
    ef_de = tiempos_d["eficiencia_DE"]
    sp_rf = tiempos_d["speedup_RF"]
    ef_rf = tiempos_d["eficiencia_RF"]

    cfg   = data["configuracion"]
    titulo = (f"Secuencial vs Paralelo (DE y RF) con checkpoints  —  "
              f"NP={cfg['NP']}, MAX_GEN={cfg['MAX_GEN']}, workers={cfg['N_WORKERS']}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(titulo, fontsize=13)

    ax = axes[0, 0]
    for res, lbl, mk in [(res_seq, "Secuencial", "o-"),
                          (res_par_de, "Paralelo DE", "s-"),
                          (res_par_rf, "Paralelo RF", "^-")]:
        ax.plot(range(len(res["hist_mejor"])), res["hist_mejor"], mk, label=lbl, markersize=5)
    ax.set_xlabel("Generacion")
    ax.set_ylabel("Mejor fitness (validacion)")
    ax.set_title("Convergencia por generacion")
    ax.legend()

    ax = axes[0, 1]
    for res, lbl, mk in [(res_seq, "Secuencial", "o-"),
                          (res_par_de, "Paralelo DE", "s-"),
                          (res_par_rf, "Paralelo RF", "^-")]:
        ax.plot(res["hist_tiempos"], res["hist_mejor"], mk, label=lbl, markersize=5)
    ax.set_xlabel("Tiempo acumulado de computo (s)")
    ax.set_ylabel("Mejor fitness (validacion)")
    ax.set_title("Fitness vs tiempo")
    ax.legend()

    ax = axes[1, 0]
    etiquetas = ["Secuencial", "Paralelo DE", "Paralelo RF"]
    valores   = [t_seq, t_de, t_rf]
    barras    = ax.bar(etiquetas, valores, color=["#2E86AB", "#A23B72", "#F18F01"])
    for bar, v in zip(barras, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(valores) * 0.01,
                f"{v:.1f} s", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Tiempo total de computo (s)")
    ax.set_title("Tiempo total de ejecucion")

    ax = axes[1, 1]
    etiquetas_sp = ["Speedup DE", "Eficiencia DE", "Speedup RF", "Eficiencia RF"]
    valores_sp   = [sp_de, ef_de, sp_rf, ef_rf]
    colores_sp   = ["#2E86AB", "#A9D6E5", "#F18F01", "#FBC490"]
    barras_sp    = ax.bar(etiquetas_sp, valores_sp, color=colores_sp)
    for bar, v in zip(barras_sp, valores_sp):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_title(f"Speedup y eficiencia (workers={cfg['N_WORKERS']})")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    guardar(fig, "comparacion_speedup.png")


def graficas_para_worker(w, res_seq, res_de, res_rf,
                         params_seq, params_de, params_rf,
                         X_train_val, y_train_val, X_test, y_test):
    sufijo = f"_w{w}" if isinstance(w, int) else "_sec"

    fig, ax = plt.subplots(figsize=(10, 5))
    for res, lbl, mk in [(res_seq, "Secuencial", "o-"),
                          (res_de, "Paralelo DE", "s-"),
                          (res_rf, "Paralelo RF", "^-")]:
        ax.plot(range(len(res["hist_mejor"])), res["hist_mejor"], mk, label=lbl, markersize=5)
    ax.set_xlabel("Generacion")
    ax.set_ylabel("Mejor fitness")
    ax.set_title(f"Convergencia (workers={w})")
    ax.legend()
    guardar(fig, f"convergencia{sufijo}.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for res, lbl, mk in [(res_seq, "Secuencial", "o-"),
                          (res_de, "Paralelo DE", "s-"),
                          (res_rf, "Paralelo RF", "^-")]:
        ax.plot(res["hist_tiempos"], res["hist_mejor"], mk, label=lbl, markersize=5)
    ax.set_xlabel("Tiempo acumulado (s)")
    ax.set_ylabel("Mejor fitness")
    ax.set_title(f"Fitness vs tiempo (workers={w})")
    ax.legend()
    guardar(fig, f"fitness_tiempo{sufijo}.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for res, lbl, mk in [(res_seq, "Secuencial", "o-"),
                          (res_de, "Paralelo DE", "s-"),
                          (res_rf, "Paralelo RF", "^-")]:
        ax.plot(range(len(res["hist_prom"])), res["hist_prom"], mk, label=lbl, markersize=5)
    ax.set_xlabel("Generacion")
    ax.set_ylabel("Fitness promedio")
    ax.set_title(f"Fitness promedio (workers={w})")
    ax.legend()
    guardar(fig, f"fitness_promedio{sufijo}.png")

    for nombre, params, clave in [("secuencial", params_seq, "secuencial"),
                                  ("paralelo_DE", params_de, "paralelo_DE"),
                                  ("paralelo_RF", params_rf, "paralelo_RF")]:
        modelo = reconstruir_modelo(params, X_train_val, y_train_val)
        y_pred = modelo.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Clase 0", "Clase 1"])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Matriz de confusion — {nombre} (workers={w})")
        guardar(fig, f"matriz_confusion_{clave}{sufijo}.png")

    etiquetas = ["Secuencial", "Paralelo DE", "Paralelo RF"]
    metricas  = ["f1", "recall", "auc"]
    nombres_m = ["F1", "Recall", "AUC-ROC"]
    valores = {m: [] for m in metricas}
    for params in [params_seq, params_de, params_rf]:
        modelo = reconstruir_modelo(params, X_train_val, y_train_val)
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]
        valores["f1"].append(f1_score(y_test, y_pred, pos_label=1))
        valores["recall"].append(recall_score(y_test, y_pred, pos_label=1))
        valores["auc"].append(roc_auc_score(y_test, y_prob))

    x = np.arange(len(etiquetas))
    ancho = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    colores = ["#2E86AB", "#A23B72", "#F18F01"]
    for i, (m, nm) in enumerate(zip(metricas, nombres_m)):
        barras = ax.bar(x + i * ancho, valores[m], ancho, label=nm, color=colores[i], alpha=0.85)
        for bar, v in zip(barras, valores[m]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x + ancho)
    ax.set_xticklabels(etiquetas)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Valor de la metrica")
    ax.set_title(f"Metricas finales en test (workers={w})")
    ax.legend()
    guardar(fig, f"metricas_finales{sufijo}.png")

    versiones = [("secuencial", "Secuencial", res_seq),
                 ("paralelo_DE", "Paralelo DE", res_de),
                 ("paralelo_RF", "Paralelo RF", res_rf)]
    fig, ax = plt.subplots(figsize=(10, 5))
    todos_tiempos = [res["tiempos_evaluacion"] for _, _, res in versiones]
    etiquetas = [nombre for _, nombre, _ in versiones]
    ax.boxplot(todos_tiempos, labels=etiquetas, patch_artist=True,
               boxprops=dict(facecolor="#A9D6E5"),
               medianprops=dict(color="#2E86AB", linewidth=2))
    ax.set_ylabel("Tiempo de evaluacion individual (s)")
    ax.set_title(f"Distribucion de tiempos de evaluacion (workers={w})")
    guardar(fig, f"tiempos_eval{sufijo}.png")


def graficas_comparacion_todos_workers(data, X_train_val, y_train_val, X_test, y_test):
    workers_data = data["resultados_por_workers"]

    lista_workers = []
    for key, val in workers_data.items():
        if key.startswith("workers_"):
            lista_workers.append(val["n_workers"])
    lista_workers.sort()

    res_sec = workers_data["secuencial"]["resumen"]
    params_sec = decodificar(res_sec["mejor_vec"])

    paleta_workers = plt.cm.tab10(np.linspace(0, 0.9, len(lista_workers)))

    print("\nGenerando graficas de comparacion global entre todos los workers...")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(res_sec["hist_mejor"])), res_sec["hist_mejor"],
            "k--", label="Secuencial", linewidth=2, zorder=5)
    for color, w in zip(paleta_workers, lista_workers):
        key = f"workers_{w}"
        val = workers_data[key]
        res_de = val["paralelo_DE"]["resumen"]
        res_rf = val["paralelo_RF"]["resumen"]
        ax.plot(range(len(res_de["hist_mejor"])), res_de["hist_mejor"],
                "-", color=color, label=f"DE w={w}", alpha=0.8)
        ax.plot(range(len(res_rf["hist_mejor"])), res_rf["hist_mejor"],
                "--", color=color, label=f"RF w={w}", alpha=0.8)
    ax.set_xlabel("Generacion")
    ax.set_ylabel("Mejor fitness (validacion)")
    ax.set_title("Convergencia — comparacion de todos los workers")
    ax.legend(fontsize=8, ncol=2)
    guardar(fig, "todos_workers_convergencia.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(res_sec["hist_tiempos"], res_sec["hist_mejor"],
            "k--", label="Secuencial", linewidth=2, zorder=5)
    for color, w in zip(paleta_workers, lista_workers):
        key = f"workers_{w}"
        val = workers_data[key]
        res_de = val["paralelo_DE"]["resumen"]
        res_rf = val["paralelo_RF"]["resumen"]
        ax.plot(res_de["hist_tiempos"], res_de["hist_mejor"],
                "-", color=color, label=f"DE w={w}", alpha=0.8)
        ax.plot(res_rf["hist_tiempos"], res_rf["hist_mejor"],
                "--", color=color, label=f"RF w={w}", alpha=0.8)
    ax.set_xlabel("Tiempo acumulado (s)")
    ax.set_ylabel("Mejor fitness (validacion)")
    ax.set_title("Fitness vs tiempo — comparacion de todos los workers")
    ax.legend(fontsize=8, ncol=2)
    guardar(fig, "todos_workers_fitness_tiempo.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(res_sec["hist_prom"])), res_sec["hist_prom"],
            "k--", label="Secuencial", linewidth=2, zorder=5)
    for color, w in zip(paleta_workers, lista_workers):
        key = f"workers_{w}"
        val = workers_data[key]
        res_de = val["paralelo_DE"]["resumen"]
        res_rf = val["paralelo_RF"]["resumen"]
        ax.plot(range(len(res_de["hist_prom"])), res_de["hist_prom"],
                "-", color=color, label=f"DE w={w}", alpha=0.8)
        ax.plot(range(len(res_rf["hist_prom"])), res_rf["hist_prom"],
                "--", color=color, label=f"RF w={w}", alpha=0.8)
    ax.set_xlabel("Generacion")
    ax.set_ylabel("Fitness promedio")
    ax.set_title("Fitness promedio — comparacion de todos los workers")
    ax.legend(fontsize=8, ncol=2)
    guardar(fig, "todos_workers_fitness_promedio.png")

    etiquetas_x = ["Secuencial"] + [f"DE w={w}" for w in lista_workers] + [f"RF w={w}" for w in lista_workers]
    metricas_claves  = ["f1", "recall", "auc"]
    metricas_nombres = ["F1", "Recall", "AUC-ROC"]
    metricas_vals = {m: [] for m in metricas_claves}

    modelo_s = reconstruir_modelo(params_sec, X_train_val, y_train_val)
    y_pred_s = modelo_s.predict(X_test)
    y_prob_s = modelo_s.predict_proba(X_test)[:, 1]
    metricas_vals["f1"].append(f1_score(y_test, y_pred_s, pos_label=1))
    metricas_vals["recall"].append(recall_score(y_test, y_pred_s, pos_label=1))
    metricas_vals["auc"].append(roc_auc_score(y_test, y_prob_s))

    for w in lista_workers:
        key = f"workers_{w}"
        val = workers_data[key]
        p_de = decodificar(val["paralelo_DE"]["resumen"]["mejor_vec"])
        m_de = reconstruir_modelo(p_de, X_train_val, y_train_val)
        y_pred_de = m_de.predict(X_test)
        y_prob_de = m_de.predict_proba(X_test)[:, 1]
        metricas_vals["f1"].append(f1_score(y_test, y_pred_de, pos_label=1))
        metricas_vals["recall"].append(recall_score(y_test, y_pred_de, pos_label=1))
        metricas_vals["auc"].append(roc_auc_score(y_test, y_prob_de))

    for w in lista_workers:
        key = f"workers_{w}"
        val = workers_data[key]
        p_rf = decodificar(val["paralelo_RF"]["resumen"]["mejor_vec"])
        m_rf = reconstruir_modelo(p_rf, X_train_val, y_train_val)
        y_pred_rf = m_rf.predict(X_test)
        y_prob_rf = m_rf.predict_proba(X_test)[:, 1]
        metricas_vals["f1"].append(f1_score(y_test, y_pred_rf, pos_label=1))
        metricas_vals["recall"].append(recall_score(y_test, y_pred_rf, pos_label=1))
        metricas_vals["auc"].append(roc_auc_score(y_test, y_prob_rf))

    x = np.arange(len(etiquetas_x))
    ancho = 0.25
    fig, ax = plt.subplots(figsize=(max(14, len(etiquetas_x) * 1.1), 6))
    colores_m = ["#2E86AB", "#A23B72", "#F18F01"]
    for i, (m, nm) in enumerate(zip(metricas_claves, metricas_nombres)):
        barras = ax.bar(x + i * ancho, metricas_vals[m], ancho, label=nm, color=colores_m[i], alpha=0.85)
        for bar, v in zip(barras, metricas_vals[m]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=7, rotation=45)
    ax.set_xticks(x + ancho)
    ax.set_xticklabels(etiquetas_x, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Valor de la metrica")
    ax.set_title("Metricas finales en test — todas las implementaciones")
    ax.legend()
    guardar(fig, "todos_workers_metricas.png")

    mejor_f1_val  = -1
    mejor_f1_etiq = ""
    for i, etiq in enumerate(etiquetas_x):
        v = metricas_vals["f1"][i]
        if v > mejor_f1_val:
            mejor_f1_val  = v
            mejor_f1_etiq = etiq

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Mejor implementacion por metrica — todas las configuraciones", fontsize=13)
    for ax, (m, nm) in zip(axes, zip(metricas_claves, metricas_nombres)):
        vals = metricas_vals[m]
        idx_mejor = int(np.argmax(vals))
        colores_bar = ["#2E86AB" if i != idx_mejor else "#E63946" for i in range(len(etiquetas_x))]
        barras = ax.bar(etiquetas_x, vals, color=colores_bar, alpha=0.85)
        for bar, v in zip(barras, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=7, rotation=45)
        ax.set_xticks(range(len(etiquetas_x)))
        ax.set_xticklabels(etiquetas_x, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(nm)
        ax.set_title(f"{nm} — Mejor: {etiquetas_x[idx_mejor]} ({vals[idx_mejor]:.3f})")
        ax.axhline(vals[idx_mejor], color="#E63946", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    guardar(fig, "todos_workers_mejor_por_metrica.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    etiquetas_t = ["Secuencial"] + [f"DE w={w}" for w in lista_workers] + [f"RF w={w}" for w in lista_workers]
    tiempos_eval = [res_sec["tiempos_evaluacion"]]
    for w in lista_workers:
        key = f"workers_{w}"
        tiempos_eval.append(workers_data[key]["paralelo_DE"]["resumen"]["tiempos_evaluacion"])
    for w in lista_workers:
        key = f"workers_{w}"
        tiempos_eval.append(workers_data[key]["paralelo_RF"]["resumen"]["tiempos_evaluacion"])
    ax.boxplot(tiempos_eval, labels=etiquetas_t, patch_artist=True,
               boxprops=dict(facecolor="#A9D6E5"),
               medianprops=dict(color="#2E86AB", linewidth=2))
    ax.set_xticklabels(etiquetas_t, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Tiempo de evaluacion individual (s)")
    ax.set_title("Distribucion de tiempos de evaluacion — todas las implementaciones")
    plt.tight_layout()
    guardar(fig, "todos_workers_tiempos_eval.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    t_sec_total = data["tiempo_secuencial"]
    t_de_lista = []
    t_rf_lista = []
    for w in lista_workers:
        key = f"workers_{w}"
        val = workers_data[key]
        t_de_lista.append(val["paralelo_DE"]["tiempo_total"])
        t_rf_lista.append(val["paralelo_RF"]["tiempo_total"])

    x_w = np.arange(len(lista_workers))
    ancho_b = 0.3
    ax.axhline(t_sec_total, color="black", linestyle="--", linewidth=1.5, label=f"Secuencial ({t_sec_total:.1f}s)")
    barras_de = ax.bar(x_w - ancho_b / 2, t_de_lista, ancho_b, color="#A23B72", alpha=0.85, label="Paralelo DE")
    barras_rf = ax.bar(x_w + ancho_b / 2, t_rf_lista, ancho_b, color="#F18F01", alpha=0.85, label="Paralelo RF")
    for bar, v in zip(barras_de, t_de_lista):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + t_sec_total * 0.01,
                f"{v:.1f}s", ha="center", fontsize=8)
    for bar, v in zip(barras_rf, t_rf_lista):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + t_sec_total * 0.01,
                f"{v:.1f}s", ha="center", fontsize=8)
    ax.set_xticks(x_w)
    ax.set_xticklabels([f"w={w}" for w in lista_workers])
    ax.set_xlabel("Numero de workers")
    ax.set_ylabel("Tiempo total (s)")
    ax.set_title("Tiempo total de ejecucion — todas las implementaciones")
    ax.legend()
    guardar(fig, "todos_workers_tiempo_total.png")


def main():
    print(f"Leyendo resultados desde '{CARPETA_SALIDA}'...")
    data = cargar_resultados()
    X_train_val, y_train_val, X_test, y_test = cargar_datos()

    if "resultados_por_workers" in data:
        print("Detectado formato multi-worker.")
        t_seq = data["tiempo_secuencial"]
        workers_data = data["resultados_por_workers"]
        grafica_speedup_vs_workers(data, t_seq)

        for key, val in workers_data.items():
            if key == "secuencial":
                continue
            w = val["n_workers"]
            print(f"\nGenerando graficas detalladas para workers = {w} ...")
            res_seq = workers_data["secuencial"]["resumen"]
            res_de  = val["paralelo_DE"]["resumen"]
            res_rf  = val["paralelo_RF"]["resumen"]
            params_seq = decodificar(workers_data["secuencial"]["resumen"]["mejor_vec"])
            params_de  = decodificar(res_de["mejor_vec"])
            params_rf  = decodificar(res_rf["mejor_vec"])
            graficas_para_worker(w, res_seq, res_de, res_rf,
                                 params_seq, params_de, params_rf,
                                 X_train_val, y_train_val, X_test, y_test)

        graficas_comparacion_todos_workers(data, X_train_val, y_train_val, X_test, y_test)

    else:
        print("Detectado formato antiguo (tres versiones).")
        grafica_comparacion_general(data)
        versiones = [("secuencial", "Secuencial"),
                     ("paralelo_DE", "Paralelo DE"),
                     ("paralelo_RF", "Paralelo RF")]
        for clave, nombre in versiones:
            params = data["evaluacion_final"][clave]["params"]
            modelo = reconstruir_modelo(params, X_train_val, y_train_val)
            y_pred = modelo.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Clase 0", "Clase 1"])
            fig, ax = plt.subplots(figsize=(6, 5))
            disp.plot(ax=ax, cmap="Blues", values_format="d")
            ax.set_title(f"Matriz de confusion — {nombre}")
            guardar(fig, f"matriz_confusion_{clave}.png")

        etiquetas = ["Secuencial", "Paralelo DE", "Paralelo RF"]
        metricas  = ["f1", "recall", "auc"]
        nombres_m = ["F1", "Recall", "AUC-ROC"]
        valores = {m: [data["evaluacion_final"][v][m] for v in ["secuencial", "paralelo_DE", "paralelo_RF"]] for m in metricas}
        x = np.arange(len(etiquetas))
        ancho = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))
        colores = ["#2E86AB", "#A23B72", "#F18F01"]
        for i, (m, nm) in enumerate(zip(metricas, nombres_m)):
            barras = ax.bar(x + i * ancho, valores[m], ancho, label=nm, color=colores[i], alpha=0.85)
            for bar, v in zip(barras, valores[m]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x + ancho)
        ax.set_xticklabels(etiquetas)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Valor de la metrica")
        ax.set_title("Metricas finales en test")
        ax.legend()
        guardar(fig, "metricas_finales.png")

        versiones = [("secuencial", "Secuencial"),
                     ("paralelo_DE", "Paralelo DE"),
                     ("paralelo_RF", "Paralelo RF")]
        fig, ax = plt.subplots(figsize=(10, 5))
        todos_tiempos = [data[clave]["tiempos_evaluacion"] for clave, _ in versiones]
        etiquetas = [nombre for _, nombre in versiones]
        ax.boxplot(todos_tiempos, labels=etiquetas, patch_artist=True,
                   boxprops=dict(facecolor="#A9D6E5"),
                   medianprops=dict(color="#2E86AB", linewidth=2))
        ax.set_ylabel("Tiempo de evaluacion individual (s)")
        ax.set_title("Distribucion de tiempos de evaluacion por version")
        guardar(fig, "historial_tiempos_eval.png")

        res_seq = data["secuencial"]
        res_par_de = data["paralelo_DE"]
        res_par_rf = data["paralelo_RF"]
        fig, ax = plt.subplots(figsize=(10, 5))
        for res, lbl, mk in [(res_seq, "Secuencial", "o-"),
                              (res_par_de, "Paralelo DE", "s-"),
                              (res_par_rf, "Paralelo RF", "^-")]:
            ax.plot(range(len(res["hist_prom"])), res["hist_prom"], mk, label=lbl, markersize=5)
        ax.set_xlabel("Generacion")
        ax.set_ylabel("Fitness promedio")
        ax.set_title("Evolucion del fitness promedio")
        ax.legend()
        guardar(fig, "fitness_promedio.png")

    print(f"\nListo. Todas las graficas estan en '{CARPETA_SALIDA}'.")


if __name__ == "__main__":
    main()