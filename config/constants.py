"""Constants and configuration for dissolution comparison app."""

from __future__ import annotations

from typing import Dict, List, Tuple

# Method names and descriptions
METHODS = {
    "f1_f2": "Factores f1/f2 (modelo-independiente)",
    "multivariate": "Región de confianza multivariante (Hotelling T²)",
    "model_dependent": "Dependiente de modelo (parámetros + Hotelling T²)",
}

# Model names
MODELS = {
    "weibull": "Weibull",
    "logistic": "Logístico", 
    "linear": "Lineal",
}

# Criteria thresholds
F1_THRESHOLD = 15.0
F2_THRESHOLD = 50.0
CONFIDENCE_LEVEL = 0.90

# File extensions
SUPPORTED_EXTENSIONS = [".xlsx", ".xls", ".csv"]

# Default metadata
DEFAULT_MEDIUM = "pH 1.2 + pepsina"

# Report formatting
MAX_TABLE_ROWS = 40
FIGURE_DPI = 200
SANITIZE_MAX_LENGTH = 24

# PDF formatting
PDF_MARGIN = 1.2  # cm
PDF_IMAGE_HEIGHT = 8.5  # cm
PDF_IMAGE_WIDTH = 16.0  # cm

# Colors for plots
COLORS = {
    "reference": "#1f77b4",
    "test": "#ff7f0e",
    "reference_fill": "#1f77b433",
    "test_fill": "#ff7f0e33",
}

# Plot styling
PLOT_STYLE = {
    "figure.figsize": (10, 6),
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
}

# Method help templates
METHOD_HELP = {
    "f1_f2": {
        "title": "A) Factores **f1 / f2** (modelo-independiente)",
        "description": "Compara las curvas usando los **promedios** por tiempo de referencia (R) y prueba (T).",
        "criteria": "- **Similares** si `f1 ≤ 15` **y** `f2 ≥ 50`.",
        "formulas": [
            r"f_1 = \frac{\sum_{t=1}^{n}\left|R_t - T_t\right|}{\sum_{t=1}^{n}R_t}\times 100",
            r"f_2 = 50\log_{10}\left(\left[1+\frac{1}{n}\sum_{t=1}^{n}(R_t-T_t)^2\right]^{-0.5}\times 100\right)"
        ]
    },
    "multivariate": {
        "title": "B) Región de confianza multivariante (**Hotelling T²**)",
        "description": "Compara el **vector** de medias a todos los tiempos a la vez, considerando la **covarianza** entre tiempos.",
        "criteria": "- **Similares** si `T² ≤ Límite crítico (90%)`.",
        "formulas": [
            r"\Delta = \bar{T} - \bar{R}",
            r"T^2 \propto \Delta^\top S_p^{-1}\Delta"
        ]
    },
    "model_dependent": {
        "title": "C) Dependiente de modelo (ajuste por unidad + comparación de parámetros)",
        "description": "1) Ajusta un modelo de 3 parámetros a cada unidad. 2) Obtiene un vector de parámetros por unidad. 3) Compara los vectores promedio entre lotes con **Hotelling T²** en el espacio de parámetros.",
        "criteria": "- **Similares** si `T²(parámetros) ≤ Límite crítico (90%)`.",
        "formulas": []
    }
}

# General conditions checklist
GENERAL_CONDITIONS = [
    "Los **tiempos de muestreo** deben ser **idénticos** para referencia y prueba.",
    "Idealmente se usan **12 unidades** por lote (o el número definido por tu protocolo).",
    "Usar **≥ 3** puntos de muestreo (mejor 4 o más).",
    "Evitar muchos puntos cuando ambos perfiles ya están cerca del **plateau** (p. ej., >85%).",
    "Si hay **alta variabilidad** (CV alto), considera el método multivariante o dependiente de modelo.",
]

# Error messages
ERROR_MESSAGES = {
    "invalid_file": "No pude leer o interpretar los archivos: {}",
    "time_mismatch": "Los tiempos de muestreo deben ser exactamente iguales en ambos archivos.",
    "missing_lots": "Ingresa el lote de referencia y el lote de prueba/test para continuar.",
    "no_files": "Carga los archivos de referencia y prueba para comenzar.",
    "invalid_format": "Formato recomendado: una fila de encabezados y columnas de tiempo como 15,20,30,45,60. Cada fila = una tableta.",
    "time_inference": "No pude inferir los tiempos. Usa encabezados numéricos para los tiempos (15,20,30...) o coloca los tiempos en la primera fila.",
}

# Success messages
SUCCESS_MESSAGES = {
    "similar": "SIMILARES",
    "not_similar": "NO SIMILARES",
}
