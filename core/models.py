"""Mathematical models for dissolution profile fitting."""

from __future__ import annotations

import numpy as np
from typing import Callable, Tuple, Dict


def logistic_model(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Logistic growth model for dissolution profiles.
    
    Args:
        t: Time points
        a: Maximum dissolution (asymptote)
        b: Growth rate parameter
        c: Inflection point (time at 50% dissolution)
    
    Returns:
        Predicted dissolution values
    """
    return a / (1 + np.exp(-b * (t - c)))


def weibull_model(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Weibull model for dissolution profiles.
    
    Args:
        t: Time points
        a: Maximum dissolution (scale parameter)
        b: Shape parameter
        c: Lag time parameter
    
    Returns:
        Predicted dissolution values
    """
    t_shift = np.maximum(t - c, 0)
    return a * (1 - np.exp(-(t_shift / (b + 1e-9))))


def linear_model(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Linear model with saturating component for dissolution profiles.
    
    Args:
        t: Time points
        a: Linear coefficient
        b: Saturating component amplitude
        c: Time constant for saturation
    
    Returns:
        Predicted dissolution values
    """
    return a * t + b * (1 - np.exp(-t / (c + 1e-9)))


# Model registry
MODEL_FUNCTIONS: Dict[str, Callable[[np.ndarray, float, float, float], np.ndarray]] = {
    "logistic": logistic_model,
    "weibull": weibull_model,
    "linear": linear_model,
}


def get_model_function(model_name: str) -> Callable[[np.ndarray, float, float, float], np.ndarray]:
    """Get model function by name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model function
        
    Raises:
        ValueError: If model name is not recognized
    """
    if model_name not in MODEL_FUNCTIONS:
        available = ", ".join(MODEL_FUNCTIONS.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    return MODEL_FUNCTIONS[model_name]


def get_model_initial_params(
    model_name: str, 
    time_points: np.ndarray, 
    unit_values: np.ndarray
) -> Tuple[float, float, float]:
    """Get reasonable initial parameters for model fitting.
    
    Args:
        model_name: Name of the model
        time_points: Array of time points
        unit_values: Array of dissolution values
        
    Returns:
        Initial parameters (a, b, c)
    """
    a0 = float(np.nanmax(unit_values))  # Maximum dissolution as initial a
    b0 = 1.0  # Default growth/shape parameter
    c0 = float(time_points[len(time_points) // 2])  # Middle time point as initial c
    
    return a0, b0, c0
