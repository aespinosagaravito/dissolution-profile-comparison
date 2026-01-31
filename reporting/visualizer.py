"""Visualization module for dissolution profiles and plots."""

from __future__ import annotations

import io
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from config.constants import COLORS, PLOT_STYLE


def setup_plot_style():
    """Configure matplotlib plot style."""
    plt.style.use('default')
    plt.rcParams.update(PLOT_STYLE)


def fig_to_png_bytes(fig) -> bytes:
    """Convert matplotlib figure to PNG bytes.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        PNG image as bytes
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def make_profile_plot(
    time_points: np.ndarray,
    ref_mean: np.ndarray,
    test_mean: np.ndarray,
    ref_label: str,
    test_label: str,
    title: str,
    ref_sd: Optional[np.ndarray] = None,
    test_sd: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Create dissolution profile comparison plot.
    
    Args:
        time_points: Array of time points
        ref_mean: Reference mean dissolution values
        test_mean: Test mean dissolution values
        ref_label: Label for reference data
        test_label: Label for test data
        title: Plot title
        ref_sd: Reference standard deviation (optional)
        test_sd: Test standard deviation (optional)
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot mean lines
    ax.plot(time_points, ref_mean, marker="o", label=ref_label, 
            color=COLORS["reference"], linewidth=2)
    ax.plot(time_points, test_mean, marker="s", label=test_label,
            color=COLORS["test"], linewidth=2)
    
    # Add standard deviation bands if provided
    if ref_sd is not None:
        ax.fill_between(time_points, ref_mean - ref_sd, ref_mean + ref_sd, 
                        alpha=0.2, color=COLORS["reference_fill"])
    if test_sd is not None:
        ax.fill_between(time_points, test_mean - test_sd, test_mean + test_sd,
                        alpha=0.2, color=COLORS["test_fill"])
    
    ax.set_xlabel("Tiempo (min)", fontsize=12)
    ax.set_ylabel("% Disuelto", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y-axis limits
    all_values = np.concatenate([ref_mean, test_mean])
    if ref_sd is not None:
        all_values = np.concatenate([all_values, ref_mean + ref_sd, ref_mean - ref_sd])
    if test_sd is not None:
        all_values = np.concatenate([all_values, test_mean + test_sd, test_mean - test_sd])
    
    y_min, y_max = np.nanmin(all_values), np.nanmax(all_values)
    ax.set_ylim(max(0, y_min - 5), min(105, y_max + 5))
    
    return fig


def make_model_fit_plot(
    time_points: np.ndarray,
    ref_mean: np.ndarray,
    test_mean: np.ndarray,
    ref_fitted: np.ndarray,
    test_fitted: np.ndarray,
    model_name: str,
    ref_lot: str,
    test_lot: str,
) -> plt.Figure:
    """Create model fitting comparison plot.
    
    Args:
        time_points: Array of time points
        ref_mean: Reference mean dissolution values
        test_mean: Test mean dissolution values
        ref_fitted: Reference fitted values
        test_fitted: Test fitted values
        model_name: Name of the fitted model
        ref_lot: Reference lot identifier
        test_lot: Test lot identifier
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ref_short = f"Ref({ref_lot})"
    test_short = f"Test({test_lot})"
    
    # Plot scatter points for actual data
    ax.scatter(time_points, ref_mean, label=f"{ref_short} (media)", 
               color=COLORS["reference"], s=60, zorder=5)
    ax.scatter(time_points, test_mean, label=f"{test_short} (media)",
               color=COLORS["test"], s=60, zorder=5)
    
    # Plot fitted curves
    ax.plot(time_points, ref_fitted, label=f"{ref_short} ajustada ({model_name})",
            color=COLORS["reference"], linewidth=2, linestyle="--")
    ax.plot(time_points, test_fitted, label=f"{test_short} ajustada ({model_name})",
            color=COLORS["test"], linewidth=2, linestyle="--")
    
    ax.set_xlabel("Tiempo (min)", fontsize=12)
    ax.set_ylabel("% Disuelto", fontsize=12)
    ax.set_title(f"Ajuste del modelo a la media | {ref_lot} vs {test_lot}", 
                fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y-axis limits
    all_values = np.concatenate([ref_mean, test_mean, ref_fitted, test_fitted])
    y_min, y_max = np.nanmin(all_values), np.nanmax(all_values)
    ax.set_ylim(max(0, y_min - 5), min(105, y_max + 5))
    
    return fig


def create_residuals_plot(
    time_points: np.ndarray,
    ref_mean: np.ndarray,
    test_mean: np.ndarray,
    ref_fitted: np.ndarray,
    test_fitted: np.ndarray,
    ref_lot: str,
    test_lot: str,
) -> plt.Figure:
    """Create residuals plot for model fitting assessment.
    
    Args:
        time_points: Array of time points
        ref_mean: Reference mean dissolution values
        test_mean: Test mean dissolution values
        ref_fitted: Reference fitted values
        test_fitted: Test fitted values
        ref_lot: Reference lot identifier
        test_lot: Test lot identifier
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ref_short = f"Ref({ref_lot})"
    test_short = f"Test({test_lot})"
    
    # Calculate residuals
    ref_residuals = ref_mean - ref_fitted
    test_residuals = test_mean - test_fitted
    
    # Plot residuals for reference
    ax1.scatter(time_points, ref_residuals, color=COLORS["reference"], s=40)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_ylabel("Residuos", fontsize=10)
    ax1.set_title(f"Residuos - {ref_short}", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot residuals for test
    ax2.scatter(time_points, test_residuals, color=COLORS["test"], s=40)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel("Tiempo (min)", fontsize=12)
    ax2.set_ylabel("Residuos", fontsize=10)
    ax2.set_title(f"Residuos - {test_short}", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
