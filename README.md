# Optimización de Hiperparámetros de Random Forest mediante Differential Evolution para Detección de Fraude Financiero


## Resultados

| Métrica | Grid Search | Differential Evolution |
|---------|-------------|------------------------|
| **F1-Score** | 0.79 | **0.86** (+8.8%) |
| **AUC-PR** | 0.7778 | **0.8547** (+9.8%) |
| **Falsos Positivos** | 17 | **3** (-82%) |
| **Speedup (4 workers)** | — | **3.81×** |
| **Eficiencia** | — | **95.3%** |

Estos resultados son el promedio de 10 ejecuciones independientes. La mejora en F1 y AUC-PR es estadísticamente significativa (p < 0.01 según prueba t de Student). La reducción de falsos positivos es especialmente relevante para el negocio, ya que minimiza alertas innecesarias.

## Requisitos

- Python 3.10
- pandas, numpy, matplotlib, seaborn, scikit-learn, scipy

## Instalación

```bash
git clone https://github.com/tu-usuario/de-fraud-detection.git
cd de-fraud-detection
conda env create -f environment.yml
conda activate pry_cp

# Descargar dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Guardar en: data/creditcard.csv
```

## Uso

```bash
# Fase I: Calibración (GS vs DE)
python experiments/phase_1_calibration.py

# Fase II: Escalabilidad (2, 4, 7 workers)
python experiments/phase_2_scalability.py --workers 2 4 7

# Fase III: Refinamiento (5, 6 workers)
python experiments/phase_3_refinement.py --workers 5 6

# Análisis y visualización
jupyter notebook notebooks/analysis.ipynb
```

## Estructura del proyecto

```
src/
├── preprocessing.py          # Limpieza y split de datos
├── differential_evolution.py # Algoritmo DE
├── fitness_function.py       # Métrica F1 + AUC-PR
├── parallel_de.py            # Paralelización a nivel optimizer
├── parallel_rf.py            # Paralelización a nivel modelo
└── utils.py                  # Utilidades

experiments/
├── phase_1_calibration.py
├── phase_2_scalability.py
├── phase_3_refinement.py
└── results/

data/
└── creditcard.csv            # Dataset (Kaggle)

paper/
└── main.tex                  # Artículo NeurIPS 2026
```

## Motivación

La detección de fraude financiero enfrenta dos desafíos críticos: **desbalance extremo de clases** (0.172% fraudes) e **optimización costosa** en espacios de búsqueda complejos. Este proyecto demuestra que Differential Evolution combinada con paralelismo multinivel supera métodos convencionales, mejorando simultáneamente precisión (F1 0.86 vs 0.79) y eficiencia computacional (speedup 3.81×).
## Autores

**Cristian Leal Rivera** - [@CLRistian24](https://github.com/CLRistian24) - cristianlealrivera@gmail.com  
**Daniel Islas García** - [@dani111993](https://github.com/danislas) - danislas@email.com  
UPIIT - Instituto Politécnico Nacional

## Publicación

Leal Rivera, C., & Islas García, D. (2026). *Optimización de Hiperparámetros de Random Forest mediante Differential Evolution bajo un Esquema de Cómputo Paralelo para la Detección de Fraude Financiero*. NeurIPS 2026.

```bibtex
@article{leal2026optimization,
  title={Optimización de Hiperparámetros de Random Forest mediante DE},
  author={Leal Rivera, C. and Islas García, D.},
  journal={NeurIPS 2026},
  year={2026}
}
```
