"""Streamlit UI module for dissolution comparison application."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit

from config.constants import (
    METHODS, MODELS, METHOD_HELP, GENERAL_CONDITIONS,
    ERROR_MESSAGES, F1_THRESHOLD, F2_THRESHOLD
)
from core.calculations import (
    compute_factors, hotelling_t2, compare_model_parameters,
    evaluate_similarity_f1_f2, evaluate_similarity_hotelling
)
from core.models import get_model_function
from data.processor import (
    read_uploaded_file, extract_time_points_and_units,
    validate_time_consistency, calculate_summary_stats
)
from reporting.visualizer import (
    make_profile_plot, make_model_fit_plot, fig_to_png_bytes
)
from reporting.generator import (
    build_pdf_report, build_excel_report, create_metadata_dict,
    generate_filename
)

# --- Visual identity (corporate palette & theming) ---
PRIMARY = "#009688"
PRIMARY_DARK = "#007c70"
NAVY = "#0c2c6a"
BORDER = "#d7e3ea"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&family=Manrope:wght@400;500;600&display=swap');

:root {{
    --primary: {PRIMARY};
    --primary-dark: {PRIMARY_DARK};
    --navy: {NAVY};
    --border: {BORDER};
}}

* {{
    font-family: 'Poppins', 'Manrope', 'Segoe UI', sans-serif;
}}

html, body, .stApp {{
    background: linear-gradient(180deg, #d7e6f4 0%, #c7dbef 45%, #c0d5ea 100%);
    color: #1d2d50;
}}

div.block-container {{
    padding-top: 10px;
    padding-left: 26px;
    padding-right: 26px;
    max-width: 1200px;
}}

section[data-testid="stSidebar"] {{
    background: #f4fbfa;
    border-right: 1px solid var(--border);
}}

section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {{
    color: var(--navy);
}}

section[data-testid="stSidebar"] .stTextInput > div > div > input,
section[data-testid="stSidebar"] .stTextArea > div > textarea,
section[data-testid="stSidebar"] .stSelectbox > div > div {{
    border-radius: 10px;
    border: 1.5px solid var(--border);
}}

.stButton>button, .stDownloadButton>button {{
    background: var(--primary);
    color: #fff;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: 600;
    letter-spacing: 0.1px;
    box-shadow: 0 8px 18px rgba(0, 150, 136, 0.28);
}}

.stButton>button:hover, .stDownloadButton>button:hover {{
    background: var(--primary-dark);
}}

div[data-testid="stExpander"] {{
    border: 1px solid var(--border);
    border-radius: 12px;
    background: #ffffff;
    box-shadow: 0 8px 16px rgba(12, 44, 106, 0.04);
}}

div[data-testid="stFileUploaderDropzone"] {{
    border: 1.5px dashed var(--border);
    background: #f7fbfc;
    border-radius: 12px;
}}

div[data-testid="stMetricValue"] {{
    color: var(--primary);
}}

div[data-testid="stDataFrame"] table thead tr th {{
    background: #f3f7fb !important;
    color: var(--navy);
}}

.hero {{
    background: linear-gradient(120deg, var(--primary) 0%, #00a69c 60%, #00b3a4 100%);
    color: #fff;
    padding: 18px 22px;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0, 138, 130, 0.26);
}}

.hero-title {{
    font-size: 26px;
    font-weight: 600;
    letter-spacing: -0.3px;
    margin: 0;
}}

.hero-sub {{
    margin: 4px 0 0 0;
    opacity: 0.92;
    font-size: 15px;
}}

.top-logo {{
    height: 54px;
    object-fit: contain;
    filter: drop-shadow(0 6px 18px rgba(0,0,0,0.12));
}}

.badge {{
    background: rgba(255, 255, 255, 0.14);
    padding: 8px 14px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 500;
}}

.card {{
    background: #fff;
    border-radius: 14px;
    padding: 18px 18px 24px 18px;
    border: 1px solid var(--border);
    box-shadow: 0 8px 20px rgba(12, 44, 106, 0.03);
}}
</style>
"""


def inject_theme() -> None:
    """Apply corporate visual theme to the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def load_logo() -> Path | None:
    """Return the company logo path if available."""
    logo_path = Path(__file__).resolve().parent.parent / "Sofgen-Pharma.png"
    return logo_path if logo_path.exists() else None


def render_hero_banner(title: str, subtitle: str) -> None:
    """Render top hero with gradient background and corporate logo."""
    logo_path = load_logo()
    cols = st.columns([4, 1.2])
    with cols[0]:
        st.markdown(
            f"""
            <div class="hero">
                <div class="hero-title">{title}</div>
                <div class="hero-sub">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        if logo_path:
            st.image(str(logo_path), width=180, caption=None)
        else:
            st.markdown(
                """
                <div class="hero" style="display:flex;align-items:center;justify-content:center;min-height:68px;">
                    <div class="badge">Logo pendiente</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_method_help(method_key: str) -> None:
    """Show an educational explanation of the selected method.
    
    Args:
        method_key: Key of the selected method
    """
    st.subheader("Explicación del método (guía rápida)")

    st.markdown("""
Esta sección explica **qué calcula** el método seleccionado, **qué fórmula usa** y **cómo se decide**
si los perfiles son **similares** o **no similares**.  
*(Consejo: si no necesitas ver esto, apaga el switch "Mostrar explicación del método" en el sidebar.)*
""")

    with st.expander("Condiciones generales recomendadas (checklist)", expanded=False):
        for condition in GENERAL_CONDITIONS:
            st.markdown(f"- {condition}")

    if method_key not in METHOD_HELP:
        return

    help_data = METHOD_HELP[method_key]
    
    st.markdown(f"### {help_data['title']}")
    st.markdown(f"**¿Qué hace?**  \n{help_data['description']}")
    st.markdown(f"**Decisión típica:**  \n{help_data['criteria']}")

    if help_data['formulas']:
        st.markdown("**Fórmulas**")
        for formula in help_data['formulas']:
            st.latex(formula)

    # Add interpretation based on method
    if method_key == "f1_f2":
        with st.expander("Interpretación rápida", expanded=False):
            st.markdown("""
- **f2 alto** (≥ 50) indica curvas muy parecidas; **f2 bajo** sugiere diferencias relevantes.
- **f1** refleja el **error relativo promedio** entre curvas (más bajo = mejor).
- Si el perfil es muy rápido y ambos llegan a >85% temprano, se recomienda usar **solo un punto**
  después de ese umbral (según práctica regulatoria).
""")
    elif method_key == "multivariate":
        with st.expander("Interpretación rápida", expanded=False):
            st.markdown("""
- **T² pequeño** significa que, globalmente, los perfiles son cercanos considerando la variabilidad.
- Si **T² supera el límite**, la diferencia global es demasiado grande para el nivel de confianza definido.
""")
    else:  # model_dependent
        st.markdown("**Qué reporta**")
        st.markdown("""
- Tabla de parámetros por unidad (referencia y prueba)
- T² en parámetros y límite crítico (90%)
- Gráfica del ajuste del modelo a la media (para visualización)
""")
        with st.expander("Interpretación rápida", expanded=False):
            st.markdown("""
- Este método es útil cuando quieres resumir la forma completa del perfil con un modelo.
- Si el ajuste del modelo es pobre (curvas ajustadas muy alejadas de los puntos), la conclusión puede
  ser menos confiable: conviene revisar el modelo o considerar el método multivariante.
""")


def render_f1_f2_results(ref_mean: np.ndarray, test_mean: np.ndarray) -> Dict[str, pd.DataFrame]:
    """Render f1/f2 method results and return tables for report.
    
    Args:
        ref_mean: Reference mean values
        test_mean: Test mean values
        
    Returns:
        Dictionary with result tables for report
    """
    f1, f2 = compute_factors(ref_mean, test_mean)
    similar, conclusion = evaluate_similarity_f1_f2(f1, f2)

    st.subheader("Resultado f1 / f2")
    m1, m2 = st.columns(2)
    m1.metric("f1", f"{f1:.2f}")
    m2.metric("f2", f"{f2:.2f}")
    
    if similar:
        st.success(conclusion)
    else:
        st.warning(conclusion)

    df_f = pd.DataFrame({
        "Métrica": ["f1", "f2"],
        "Valor": [round(f1, 4), round(f2, 4)],
        "Criterio": [f"≤ {F1_THRESHOLD}", f"≥ {F2_THRESHOLD}"],
        "Cumple": [f1 <= F1_THRESHOLD, f2 >= F2_THRESHOLD],
    })
    
    return {"Cálculo f1/f2": df_f}


def render_multivariate_results(ref_units: np.ndarray, test_units: np.ndarray) -> Dict[str, pd.DataFrame]:
    """Render multivariate method results and return tables for report.
    
    Args:
        ref_units: Reference units matrix
        test_units: Test units matrix
        
    Returns:
        Dictionary with result tables for report
    """
    T2, df1, df2, ci90 = hotelling_t2(ref_units, test_units)
    similar, conclusion = evaluate_similarity_hotelling(T2, ci90)

    st.subheader("Resultado multivariante (Hotelling T²)")
    st.json({"T2": T2, "df1": df1, "df2": df2, "Límite 90%": ci90})
    
    if similar:
        st.success(conclusion)
    else:
        st.warning(conclusion)

    df_mv = pd.DataFrame({
        "Métrica": ["Hotelling T²", "Límite crítico (90%)", "df1", "df2"],
        "Valor": [round(T2, 6), round(ci90, 6), df1, df2],
    })
    
    return {"Comparación multivariante": df_mv}


def render_model_dependent_results(
    ref_units: np.ndarray,
    test_units: np.ndarray,
    time_points: np.ndarray,
    ref_lot: str,
    test_lot: str
) -> Dict[str, pd.DataFrame]:
    """Render model-dependent method results and return tables/figures for report.
    
    Args:
        ref_units: Reference units matrix
        test_units: Test units matrix
        time_points: Array of time points
        ref_lot: Reference lot identifier
        test_lot: Test lot identifier
        
    Returns:
        Dictionary with result tables and figures for report
    """
    model_choice = st.selectbox("Modelo", list(MODELS.values()), index=0)
    model_key = list(MODELS.keys())[list(MODELS.values()).index(model_choice)]
    
    result = compare_model_parameters(ref_units, test_units, time_points, model_key)
    T2 = result["T2"]
    ci90 = result["ci90"]
    similar, conclusion = evaluate_similarity_hotelling(T2, ci90, "(parámetros) ")

    st.subheader("Resultado dependiente de modelo")
    st.json({"Modelo": model_choice, "T2": T2, "Límite 90%": ci90})
    
    if similar:
        st.success(conclusion)
    else:
        st.warning(conclusion)

    # Create parameter tables
    ref_params = pd.DataFrame(result["ref_params"], columns=["a", "b", "c"])
    test_params = pd.DataFrame(result["test_params"], columns=["a", "b", "c"])
    ref_params.insert(0, "Unidad", np.arange(1, len(ref_params) + 1))
    test_params.insert(0, "Unidad", np.arange(1, len(test_params) + 1))

    tables = {
        f"Parámetros modelo (Referencia) - {model_choice}": ref_params.round(6),
        f"Parámetros modelo (Prueba) - {model_choice}": test_params.round(6),
        "Comparación en parámetros": pd.DataFrame({
            "Métrica": ["Hotelling T² (parámetros)", "Límite crítico (90%)", "df1", "df2"],
            "Valor": [round(T2, 6), round(ci90, 6), result["df1"], result["df2"]],
        })
    }

    # Create fitted curves plot
    ref_mean = ref_units.mean(axis=0)
    test_mean = test_units.mean(axis=0)
    model_func = get_model_function(model_key)
    
    # Fit model to mean curves
    popt_ref, _ = curve_fit(
        model_func, time_points, ref_mean,
        p0=[ref_mean.max(), 1.0, time_points[len(time_points) // 2]],
        maxfev=10000,
    )
    popt_test, _ = curve_fit(
        model_func, time_points, test_mean,
        p0=[test_mean.max(), 1.0, time_points[len(time_points) // 2]],
        maxfev=10000,
    )
    
    ref_fitted = model_func(time_points, *popt_ref)
    test_fitted = model_func(time_points, *popt_test)
    
    ref_short = f"Ref({ref_lot})"
    test_short = f"Test({test_lot})"
    
    fig_fit = make_model_fit_plot(
        time_points, ref_mean, test_mean, ref_fitted, test_fitted,
        model_choice, ref_lot, test_lot
    )
    st.pyplot(fig_fit)
    
    figures = {
        f"Ajuste del modelo a la media ({model_choice})": fig_to_png_bytes(fig_fit)
    }
    
    return tables, figures


def render_data_tables(
    df_ref_num: pd.DataFrame,
    df_test_num: pd.DataFrame,
    ref_label: str,
    test_label: str,
    show_details: bool
) -> None:
    """Render data tables in UI.
    
    Args:
        df_ref_num: Reference numeric DataFrame
        df_test_num: Test numeric DataFrame
        ref_label: Reference label
        test_label: Test label
        show_details: Whether to show details by default
    """
    if show_details:
        st.subheader("Datos (tablas numéricas usadas para el cálculo)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{ref_label}**")
            st.dataframe(df_ref_num)
        with c2:
            st.markdown(f"**{test_label}**")
            st.dataframe(df_test_num)
    else:
        with st.expander("Ver tablas numéricas usadas para el cálculo"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{ref_label}**")
                st.dataframe(df_ref_num)
            with c2:
                st.markdown(f"**{test_label}**")
                st.dataframe(df_test_num)


def render_summary_table(overview: pd.DataFrame, show_details: bool) -> None:
    """Render summary statistics table.
    
    Args:
        overview: Summary statistics DataFrame
        show_details: Whether to show details by default
    """
    if show_details:
        st.subheader("Resumen por tiempo")
        st.dataframe(overview)
    else:
        with st.expander("Ver resumen por tiempo (media, DE, CV%)"):
            st.dataframe(overview)


def render_download_buttons(
    metadata: Dict[str, str],
    method_name: str,
    conclusion: str,
    report_tables: Dict[str, pd.DataFrame],
    report_figs: Dict[str, bytes],
    ref_label: str,
    test_label: str
) -> None:
    """Render download buttons for reports.
    
    Args:
        metadata: Report metadata
        method_name: Name of method used
        conclusion: Similarity conclusion
        report_tables: Report tables
        report_figs: Report figures
        ref_label: Reference label
        test_label: Test label
    """
    st.subheader("Descargas")

    pdf_bytes = build_pdf_report(
        title="Reporte – Comparación de perfiles de disolución",
        metadata=metadata,
        method_name=method_name,
        conclusion=f"{ref_label} vs {test_label} → {conclusion}",
        tables=report_tables,
        figures=report_figs,
    )

    xlsx_bytes = build_excel_report(report_tables)

    # Generate filenames
    pdf_filename = generate_filename(
        metadata["Producto"], metadata["Lote referencia"], 
        metadata["Lote test"], "reporte", "pdf"
    )
    xlsx_filename = generate_filename(
        metadata["Producto"], metadata["Lote referencia"],
        metadata["Lote test"], "tablas", "xlsx"
    )

    st.download_button(
        "Descargar reporte PDF",
        data=pdf_bytes,
        file_name=pdf_filename,
        mime="application/pdf",
    )
    st.download_button(
        "Descargar tablas (Excel)",
        data=xlsx_bytes,
        file_name=xlsx_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# Running this module directly isn't the intended entry point.
# Show a friendly message instead of a blank page if executed with `streamlit run ui/streamlit_app.py`.
if __name__ == "__main__":
    st.set_page_config(page_title="Comparación de perfiles de disolución", layout="wide")
    inject_theme()
    st.title("Usa main.py para ejecutar la app")
    st.write(
        "Este módulo solo define los componentes de UI. "
        "Ejecuta `streamlit run main.py` para iniciar la aplicación completa."
    )
