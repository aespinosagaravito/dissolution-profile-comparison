"""Data processing module for file reading and data validation."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple

from config.constants import SUPPORTED_EXTENSIONS, ERROR_MESSAGES


def _can_float(x) -> bool:
    """Check if a value can be converted to float."""
    try:
        float(str(x).strip())
        return True
    except Exception:
        return False


def read_uploaded_file(file) -> pd.DataFrame:
    """Read uploaded file (Excel or CSV) into DataFrame.
    
    Args:
        file: Uploaded file object
        
    Returns:
        DataFrame with file contents
        
    Raises:
        ValueError: If file format is not supported
    """
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    else:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise ValueError(f"Unsupported file format. Supported formats: {supported}")


def extract_time_points_and_units(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Robust extractor of time points and unit data from DataFrame.
    
    Preferred format: time points are column headers (15,20,30,...) and each row is a unit.
    Legacy format: first row contains time points.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (time_points, units_array, cleaned_dataframe)
        
    Raises:
        ValueError: If time points cannot be inferred
    """
    # Drop fully empty rows/cols
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Preferred: headers contain numeric-like time points
    time_cols = [c for c in df.columns if _can_float(c)]
    if len(time_cols) >= 3:
        times = np.array(sorted([float(c) for c in time_cols]))
        
        # Order columns by time
        ordered_cols = []
        for t in times:
            match = None
            for c in time_cols:
                if float(str(c).strip()) == float(t):
                    match = c
                    break
            ordered_cols.append(match)

        df_numeric = df[ordered_cols].apply(pd.to_numeric, errors="coerce")
        df_numeric = df_numeric.dropna(how="all")
        units = df_numeric.dropna(axis=0, how="any").values.astype(float)
        
        return times, units, df_numeric

    # Legacy: first row contains times
    first_row = df.iloc[0]
    idxs = [i for i, v in enumerate(first_row.values) if _can_float(v)]
    if len(idxs) >= 3:
        times = first_row.iloc[idxs].astype(float).values
        df_numeric = df.iloc[1:, idxs].apply(pd.to_numeric, errors="coerce")
        df_numeric = df_numeric.dropna(how="all")
        units = df_numeric.dropna(axis=0, how="any").values.astype(float)
        
        return times, units, df_numeric

    raise ValueError(ERROR_MESSAGES["time_inference"])


def validate_time_consistency(time_ref: np.ndarray, time_test: np.ndarray) -> bool:
    """Validate that time points are consistent between reference and test.
    
    Args:
        time_ref: Reference time points
        time_test: Test time points
        
    Returns:
        True if times are consistent, False otherwise
    """
    return time_ref.size == time_test.size and np.allclose(time_ref, time_test)


def calculate_summary_stats(
    ref_units: np.ndarray, 
    test_units: np.ndarray, 
    time_points: np.ndarray,
    ref_lot: str,
    test_lot: str
) -> pd.DataFrame:
    """Calculate summary statistics for both datasets.
    
    Args:
        ref_units: Reference units matrix
        test_units: Test units matrix
        time_points: Array of time points
        ref_lot: Reference lot identifier
        test_lot: Test lot identifier
        
    Returns:
        DataFrame with summary statistics
    """
    ref_mean = ref_units.mean(axis=0)
    test_mean = test_units.mean(axis=0)
    ref_sd = ref_units.std(axis=0, ddof=1)
    test_sd = test_units.std(axis=0, ddof=1)
    
    ref_short = f"Ref({ref_lot})"
    test_short = f"Test({test_lot})"
    
    overview = pd.DataFrame({
        "Tiempo (min)": time_points,
        f"{ref_short} media": np.round(ref_mean, 4),
        f"{ref_short} DE": np.round(ref_sd, 4),
        f"{ref_short} CV%": np.round((ref_sd / np.maximum(ref_mean, 1e-9)) * 100, 2),
        f"{test_short} media": np.round(test_mean, 4),
        f"{test_short} DE": np.round(test_sd, 4),
        f"{test_short} CV%": np.round((test_sd / np.maximum(test_mean, 1e-9)) * 100, 2),
    })
    
    return overview


def sanitize_filename_token(s: str, max_len: int = 24) -> str:
    """Sanitize string for use in filename.
    
    Args:
        s: Input string
        max_len: Maximum length
        
    Returns:
        Sanitized string
    """
    import re
    
    s2 = re.sub(r"[^A-Za-z0-9_-]+", "_", (s or "").strip())
    s2 = re.sub(r"_+", "_", s2).strip("_")
    return (s2[:max_len] or "NA")
