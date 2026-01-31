"""Core calculation engine for dissolution comparison methods."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import f as f_dist
from typing import Callable, Dict, Tuple

from config.constants import CONFIDENCE_LEVEL
from core.models import get_model_function, get_model_initial_params


def compute_factors(reference: np.ndarray, test: np.ndarray) -> Tuple[float, float]:
    """Compute difference factor (f1) and similarity factor (f2).
    
    Args:
        reference: Reference dissolution values
        test: Test dissolution values
        
    Returns:
        Tuple of (f1, f2) factors
        
    Raises:
        ValueError: If arrays have different shapes
    """
    if reference.shape != test.shape:
        raise ValueError("Reference and test arrays must be the same length")
    
    n = reference.size
    f1 = (np.sum(np.abs(reference - test)) / np.sum(reference)) * 100
    f2 = 50 * np.log10((1 + (1 / n) * np.sum(np.square(reference - test))) ** -0.5 * 100)
    
    return float(f1), float(f2)


def hotelling_t2(ref_units: np.ndarray, test_units: np.ndarray) -> Tuple[float, int, int, float]:
    """Hotelling T² with a 90% critical limit based on an F distribution.
    
    Args:
        ref_units: Reference units matrix (n_units x n_timepoints)
        test_units: Test units matrix (n_units x n_timepoints)
        
    Returns:
        Tuple of (T2_statistic, df1, df2, critical_value)
        
    Raises:
        ValueError: If matrices have different shapes
    """
    if ref_units.shape != test_units.shape:
        raise ValueError("Reference and test matrices must have the same shape")

    m, k = ref_units.shape
    ref_mean = ref_units.mean(axis=0)
    test_mean = test_units.mean(axis=0)
    diff = test_mean - ref_mean

    # Calculate pooled covariance matrix
    cov_ref = np.cov(ref_units, rowvar=False, ddof=1)
    cov_test = np.cov(test_units, rowvar=False, ddof=1)
    pooled_cov = ((m - 1) * cov_ref + (m - 1) * cov_test) / (2 * m - 2)
    
    # Regularization to ensure invertibility
    reg_cov = pooled_cov + np.eye(k) * 1e-6
    inv_cov = np.linalg.inv(reg_cov)

    # Calculate T² statistic
    T2 = (m * m) / (2 * m) * (diff @ inv_cov @ diff)
    
    # Degrees of freedom
    df1 = k
    df2 = 2 * m - k - 1
    
    # Critical value at specified confidence level
    F_crit = f_dist.ppf(CONFIDENCE_LEVEL, df1, df2)
    ci90 = (df1 * (2 * m - 1) / df2) * F_crit
    
    return float(T2), int(df1), int(df2), float(ci90)


def fit_unit_model(
    time_points: np.ndarray,
    unit_values: np.ndarray,
    model_func: Callable[[np.ndarray, float, float, float], np.ndarray],
) -> Tuple[float, float, float]:
    """Fit a 3-parameter model to a single unit's dissolution data.
    
    Args:
        time_points: Array of time points
        unit_values: Dissolution values for a single unit
        model_func: Model function to fit
        
    Returns:
        Tuple of fitted parameters (a, b, c)
    """
    a0, b0, c0 = get_model_initial_params("", time_points, unit_values)
    popt, _ = curve_fit(
        model_func, 
        time_points, 
        unit_values, 
        p0=[a0, b0, c0], 
        maxfev=10000
    )
    return float(popt[0]), float(popt[1]), float(popt[2])


def compare_model_parameters(
    ref_units: np.ndarray,
    test_units: np.ndarray,
    time_points: np.ndarray,
    model_name: str,
) -> Dict[str, object]:
    """Fit model per unit; compare 3-parameter vectors using Hotelling T².
    
    Args:
        ref_units: Reference units matrix
        test_units: Test units matrix
        time_points: Array of time points
        model_name: Name of the model to fit
        
    Returns:
        Dictionary with comparison results
        
    Raises:
        ValueError: If matrices have different shapes
    """
    if ref_units.shape != test_units.shape:
        raise ValueError("Reference and test matrices must have the same shape")

    model_func = get_model_function(model_name)
    m, _ = ref_units.shape
    
    # Fit model to each unit
    ref_params = np.array([
        fit_unit_model(time_points, ref_units[i], model_func) 
        for i in range(m)
    ])
    test_params = np.array([
        fit_unit_model(time_points, test_units[i], model_func) 
        for i in range(m)
    ])

    # Hotelling T² in parameter space
    ref_mean = ref_params.mean(axis=0)
    test_mean = test_params.mean(axis=0)
    diff = test_mean - ref_mean

    cov_ref = np.cov(ref_params, rowvar=False, ddof=1)
    cov_test = np.cov(test_params, rowvar=False, ddof=1)
    pooled_cov = ((m - 1) * cov_ref + (m - 1) * cov_test) / (2 * m - 2)
    reg_cov = pooled_cov + np.eye(3) * 1e-6
    inv_cov = np.linalg.inv(reg_cov)

    T2 = (m * m) / (2 * m) * (diff @ inv_cov @ diff)
    df1 = 3
    df2 = 2 * m - df1 - 1
    F_crit = f_dist.ppf(CONFIDENCE_LEVEL, df1, df2)
    ci90 = (df1 * (2 * m - 1) / df2) * F_crit
    similar = bool(T2 <= ci90)

    return {
        "ref_params": ref_params,
        "test_params": test_params,
        "T2": float(T2),
        "df1": int(df1),
        "df2": int(df2),
        "ci90": float(ci90),
        "similar": similar,
    }


def evaluate_similarity_f1_f2(f1: float, f2: float) -> Tuple[bool, str]:
    """Evaluate similarity based on f1/f2 criteria.
    
    Args:
        f1: Difference factor
        f2: Similarity factor
        
    Returns:
        Tuple of (is_similar, conclusion_text)
    """
    from config.constants import F1_THRESHOLD, F2_THRESHOLD, SUCCESS_MESSAGES
    
    similar = (f1 <= F1_THRESHOLD) and (f2 >= F2_THRESHOLD)
    conclusion = f"{'SIMILARES' if similar else 'NO SIMILARES'} (criterio típico: f1 ≤ {F1_THRESHOLD} y f2 ≥ {F2_THRESHOLD})."
    
    return similar, conclusion


def evaluate_similarity_hotelling(T2: float, ci90: float, context: str = "") -> Tuple[bool, str]:
    """Evaluate similarity based on Hotelling T² criteria.
    
    Args:
        T2: Hotelling T² statistic
        ci90: Critical value at 90% confidence
        context: Additional context for conclusion text
        
    Returns:
        Tuple of (is_similar, conclusion_text)
    """
    from config.constants import SUCCESS_MESSAGES
    
    similar = T2 <= ci90
    conclusion = f"{'SIMILARES' if similar else 'NO SIMILARES'} {context}(T²={T2:.4f} vs Límite 90%={ci90:.4f})."
    
    return similar, conclusion
