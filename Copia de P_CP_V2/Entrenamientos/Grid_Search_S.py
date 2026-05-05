import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
)

# rutas de los archivos, ajustar según la ubicación real
RUTA_X_TRAIN = "Archivos/Division/X_train.csv"
RUTA_Y_TRAIN = "Archivos/Division/y_train.csv"
RUTA_X_VAL   = "Archivos/Division/X_val.csv"
RUTA_Y_VAL   = "Archivos/Division/y_val.csv"
RUTA_X_TEST  = "Archivos/Division/X_test.csv"
RUTA_Y_TEST  = "Archivos/Division/y_test.csv"

# carga de los tres conjuntos
X_train = pd.read_csv(RUTA_X_TRAIN)
y_train = pd.read_csv(RUTA_Y_TRAIN).squeeze()

X_val = pd.read_csv(RUTA_X_VAL)
y_val = pd.read_csv(RUTA_Y_VAL).squeeze()

X_test = pd.read_csv(RUTA_X_TEST)
y_test = pd.read_csv(RUTA_Y_TEST).squeeze()

# el conjunto de validacion se concatena al de entrenamiento para que GridSearchCV
# lo use directamente sin partirlo de nuevo con CV interno
X_trainval = pd.concat([X_train, X_val], ignore_index=True)
y_trainval = pd.concat([y_train, y_val], ignore_index=True)

print("Tamaños de conjuntos:")
print(f"  Entrenamiento: {X_train.shape}, fraudes={y_train.sum()}")
print(f"  Validación:    {X_val.shape},  fraudes={y_val.sum()}")
print(f"  Prueba:        {X_test.shape},  fraudes={y_test.sum()}")

# espacio de hiperparámetros según lo especificado en el trabajo
# se definen rangos discretos representativos dentro de los intervalos continuos
# para hacer la búsqueda computacionalmente manejable
param_grid = {
    "n_estimators":      [100, 300, 500, 800, 1000],
    "max_depth":         [3, 10, 20, 35, 50],
    "min_samples_split": [2, 10, 25, 50],
    "min_samples_leaf":  [1, 5, 10, 20],
    "max_features":      [0.1, 0.3, 0.5, 0.7, 1.0],
    "bootstrap":         [True, False],
    "class_weight":      [None, "balanced", "balanced_subsample"],
    "criterion":         ["gini", "entropy", "log_loss"],
}

# clasificador base sin paralelización (n_jobs=1)
rf_base = RandomForestClassifier(random_state=42, n_jobs=1)

# validación cruzada estratificada para respetar el desbalance de clases
# n_splits=5 con shuffle asegura que cada fold contenga ejemplos de fraude
cv_estratificado = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# se optimiza por average_precision (área bajo la curva PR) porque en datasets
# con desbalance extremo es más informativa que el ROC-AUC
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    scoring="average_precision",
    cv=cv_estratificado,
    refit=True,          # re-entrena el mejor modelo sobre todo X_trainval al terminar
    verbose=2,
    n_jobs=1,            # sin paralelización, como se requiere
    return_train_score=False,
)

print("\nIniciando GridSearchCV...")
tiempo_inicio_grid = time.time()
grid_search.fit(X_trainval, y_trainval)
tiempo_fin_grid = time.time()
tiempo_grid = tiempo_fin_grid - tiempo_inicio_grid

print(f"\nMejores hiperparámetros encontrados:")
for parametro, valor in grid_search.best_params_.items():
    print(f"  {parametro}: {valor}")

print(f"\nMejor average_precision en CV: {grid_search.best_score_:.4f}")
print(f"Tiempo total GridSearchCV:     {tiempo_grid:.2f} s  ({tiempo_grid/60:.2f} min)")

# el mejor modelo ya está reentrenado sobre X_trainval completo por refit=True
mejor_modelo = grid_search.best_estimator_

# medición de tiempo de inferencia sobre el conjunto de prueba
tiempo_inicio_pred = time.time()
y_pred = mejor_modelo.predict(X_test)
tiempo_fin_pred = time.time()
tiempo_pred = tiempo_fin_pred - tiempo_inicio_pred

y_prob = mejor_modelo.predict_proba(X_test)[:, 1]  # probabilidad de fraude

print("\nResultados en conjunto de prueba:")
print(classification_report(y_test, y_pred, target_names=["Legítima", "Fraude"]))

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# métricas orientadas al desbalance de clases
f1_fraude  = f1_score(y_test, y_pred, pos_label=1)
pr_auc     = average_precision_score(y_test, y_prob)
recall_pos = recall_score(y_test, y_pred, pos_label=1)  # recall de la clase positiva (fraude)

print(f"\nF1-Score clase fraude:         {f1_fraude:.4f}")
print(f"AUC-PR:                        {pr_auc:.4f}")
print(f"Recall clase positiva (fraude):{recall_pos:.4f}")
print(f"\nTiempo de inferencia (prueba): {tiempo_pred:.4f} s")

# importancia de características ordenada de mayor a menor
importancias = pd.Series(
    mejor_modelo.feature_importances_,
    index=X_test.columns
).sort_values(ascending=False)

print("\nTop 10 características más importantes:")
print(importancias.head(10).to_string())