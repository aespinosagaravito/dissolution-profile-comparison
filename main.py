"""Main application entry point for dissolution comparison system."""

from __future__ import annotations

import asyncio
import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict

# Import modular components
from config.constants import METHODS, MODELS, ERROR_MESSAGES, DEFAULT_MEDIUM
from data.processor import calculate_summary_stats
from ui.streamlit_app import (
    render_method_help, render_f1_f2_results, render_multivariate_results,
    render_model_dependent_results, render_data_tables, render_summary_table,
    render_download_buttons
)
from agents.orchestrator import AgentOrchestrator


def render_sidebar_inputs() -> Dict[str, any]:
    """Render sidebar inputs and return user selections."""
    st.sidebar.header("Entrada")
    
    # UI controls
    show_details = st.sidebar.toggle("Mostrar detalles en pantalla", value=False)
    method_display = st.sidebar.selectbox(
        "Método",
        list(METHODS.values()),
        index=0
    )
    method_key = list(METHODS.keys())[list(METHODS.values()).index(method_display)]
    
    show_math = st.sidebar.toggle("Mostrar explicación del método", value=False)
    
    # File uploads
    upload_ref = st.sidebar.file_uploader("Referencia (pre-cambio)", type=["xlsx", "xls", "csv"])
    upload_test = st.sidebar.file_uploader("Prueba (post-cambio)", type=["xlsx", "xls", "csv"])
    
    # Metadata inputs
    st.sidebar.header("Metadatos (obligatorio para trazabilidad)")
    product_name = st.sidebar.text_input("Producto / Código", value="")
    active_name = st.sidebar.text_input("Activo / Fuerza", value="")
    reference_lot = st.sidebar.text_input("Lote referencia (pre-cambio)", value="")
    test_lot = st.sidebar.text_input("Lote prueba / test (post-cambio)", value="")
    medium = st.sidebar.text_input("Medio", value=DEFAULT_MEDIUM)
    apparatus = st.sidebar.text_input("Aparato / RPM", value="")
    analyst = st.sidebar.text_input("Analista", value="")
    notes = st.sidebar.text_area("Notas", value="")
    
    return {
        "show_details": show_details,
        "method_key": method_key,
        "method_display": method_display,
        "show_math": show_math,
        "upload_ref": upload_ref,
        "upload_test": upload_test,
        "product_name": product_name,
        "active_name": active_name,
        "reference_lot": reference_lot,
        "test_lot": test_lot,
        "medium": medium,
        "apparatus": apparatus,
        "analyst": analyst,
        "notes": notes
    }


def validate_inputs(inputs: Dict[str, any]) -> bool:
    """Validate user inputs.
    
    Args:
        inputs: Dictionary of user inputs
        
    Returns:
        True if inputs are valid, False otherwise
    """
    if not (inputs["upload_ref"] and inputs["upload_test"]):
        st.info(ERROR_MESSAGES["no_files"])
        return False
    
    if not inputs["reference_lot"].strip() or not inputs["test_lot"].strip():
        st.warning(ERROR_MESSAGES["missing_lots"])
        st.stop()
        return False
    
    return True


async def run_analysis_with_agents(inputs: Dict[str, any]) -> Dict[str, any]:
    """Run analysis using agent orchestration.
    
    Args:
        inputs: Dictionary of user inputs
        
    Returns:
        Analysis results
    """
    # Create metadata
    from reporting.generator import create_metadata_dict
    
    metadata = create_metadata_dict(
        product_name=inputs["product_name"],
        active_name=inputs["active_name"],
        reference_lot=inputs["reference_lot"],
        test_lot=inputs["test_lot"],
        medium=inputs["medium"],
        apparatus=inputs["apparatus"],
        analyst=inputs["analyst"],
        notes=inputs["notes"]
    )
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    
    # Run analysis
    try:
        results = await orchestrator.execute_analysis(
            method=inputs["method_key"],
            ref_file=inputs["upload_ref"],
            test_file=inputs["upload_test"],
            metadata=metadata
        )
        
        return results
        
    except Exception as e:
        st.error(f"Error en el análisis: {str(e)}")
        return None


def run_analysis_traditional(inputs: Dict[str, any]) -> Dict[str, any]:
    """Run analysis using traditional approach (for compatibility).
    
    Args:
        inputs: Dictionary of user inputs
        
    Returns:
        Analysis results
    """
    from data.processor import read_uploaded_file, extract_time_points_and_units, validate_time_consistency
    from reporting.visualizer import make_profile_plot, fig_to_png_bytes
    from reporting.generator import create_metadata_dict
    
    try:
        # Read and process data
        df_ref_raw = read_uploaded_file(inputs["upload_ref"])
        df_test_raw = read_uploaded_file(inputs["upload_test"])
        time_ref, ref_units, df_ref_num = extract_time_points_and_units(df_ref_raw)
        time_test, test_units, df_test_num = extract_time_points_and_units(df_test_raw)
        
        # Validate time consistency
        if not validate_time_consistency(time_ref, time_test):
            st.error(ERROR_MESSAGES["time_mismatch"])
            with st.expander("Ver detalles de tiempos"):
                st.write("Tiempos referencia:", time_ref)
                st.write("Tiempos prueba:", time_test)
            return None
        
        time_points = time_ref
        
        # Calculate summary statistics
        overview = calculate_summary_stats(
            ref_units, test_units, time_points,
            inputs["reference_lot"], inputs["test_lot"]
        )
        
        # Calculate means and standard deviations
        ref_mean = ref_units.mean(axis=0)
        test_mean = test_units.mean(axis=0)
        ref_sd = ref_units.std(axis=0, ddof=1)
        test_sd = test_units.std(axis=0, ddof=1)
        
        # Create labels and title
        ref_label = f"Referencia (Lote {inputs['reference_lot']})"
        test_label = f"Test (Lote {inputs['test_lot']})"
        title_profile = "Perfil de disolución"
        if inputs["product_name"].strip():
            title_profile = f"{title_profile} – {inputs['product_name']}"
        title_profile = f"{title_profile} | {inputs['reference_lot']} vs {inputs['test_lot']}"
        
        # Generate profile plot
        fig_profile = make_profile_plot(
            time_points, ref_mean, test_mean, ref_label, test_label, title_profile, ref_sd, test_sd
        )
        
        # Prepare report objects
        report_tables = {"Resumen por tiempo": overview}
        report_figs = {"Perfil de disolución (media ± DE)": fig_to_png_bytes(fig_profile)}
        
        # Method-specific analysis
        analysis_result = None
        if inputs["method_key"] == "f1_f2":
            report_tables.update(render_f1_f2_results(ref_mean, test_mean))
        elif inputs["method_key"] == "multivariate":
            report_tables.update(render_multivariate_results(ref_units, test_units))
        else:  # model_dependent
            tables, figures = render_model_dependent_results(
                ref_units, test_units, time_points, 
                inputs["reference_lot"], inputs["test_lot"]
            )
            report_tables.update(tables)
            report_figs.update(figures)
        
        # Create metadata
        metadata = create_metadata_dict(
            product_name=inputs["product_name"],
            active_name=inputs["active_name"],
            reference_lot=inputs["reference_lot"],
            test_lot=inputs["test_lot"],
            medium=inputs["medium"],
            apparatus=inputs["apparatus"],
            analyst=inputs["analyst"],
            notes=inputs["notes"],
            n_units=ref_units.shape[0],
            time_points=time_points.tolist()
        )
        
        return {
            "data": {
                "time_points": time_points,
                "ref_units": ref_units,
                "test_units": test_units,
                "df_ref_num": df_ref_num,
                "df_test_num": df_test_num,
                "overview": overview,
                "ref_mean": ref_mean,
                "test_mean": test_mean,
                "ref_sd": ref_sd,
                "test_sd": test_sd
            },
            "report_tables": report_tables,
            "report_figs": report_figs,
            "metadata": metadata,
            "ref_label": ref_label,
            "test_label": test_label,
            "title_profile": title_profile,
            "fig_profile": fig_profile
        }
        
    except Exception as e:
        st.error(ERROR_MESSAGES["invalid_file"].format(str(e)))
        st.info(ERROR_MESSAGES["invalid_format"])
        return None


def main() -> None:
    """Main application entry point."""
    st.set_page_config(page_title="Comparación de perfiles de disolución", layout="wide")
    st.title("Comparación de perfiles de disolución (FDA IR) + Reporte")
    
    # Get user inputs
    inputs = render_sidebar_inputs()
    
    # Validate inputs
    if not validate_inputs(inputs):
        return
    
    # Show method help if requested
    if inputs["show_math"]:
        render_method_help(inputs["method_key"])
        st.divider()
    
    # Choose execution mode
    use_agents = st.sidebar.checkbox("Usar arquitectura de agentes (experimental)", value=False)
    
    if use_agents:
        # Run with agent orchestration
        if st.sidebar.button("Ejecutar análisis con agentes"):
            with st.spinner("Ejecutando análisis con agentes..."):
                results = asyncio.run(run_analysis_with_agents(inputs))
                
                if results:
                    st.success("Análisis completado con agentes")
                    st.json(results["execution_summary"])
                    
                    # Display results (simplified for agent mode)
                    if results["analysis"]:
                        st.subheader("Resultados del análisis")
                        st.json(results["analysis"])
                    
                    # Render downloads if available
                    if results["reports"]:
                        render_download_buttons(
                            metadata={},  # Would be passed from agent results
                            method_name=inputs["method_display"],
                            conclusion=results["analysis"]["conclusion"] if results["analysis"] else "No conclusion",
                            report_tables={},
                            report_figs={},
                            ref_label=inputs["reference_lot"],
                            test_label=inputs["test_lot"]
                        )
    else:
        # Run traditional analysis
        results = run_analysis_traditional(inputs)
        
        if results is None:
            return
        
        # Display data tables
        render_data_tables(
            results["df_ref_num"], results["df_test_num"],
            results["ref_label"], results["test_label"], inputs["show_details"]
        )
        
        # Display summary table
        render_summary_table(results["overview"], inputs["show_details"])
        
        # Display profile plot
        st.pyplot(results["fig_profile"])
        
        # Display downloads
        render_download_buttons(
            metadata=results["metadata"],
            method_name=inputs["method_display"],
            conclusion="",  # Would be extracted from analysis results
            report_tables=results["report_tables"],
            report_figs=results["report_figs"],
            ref_label=results["ref_label"],
            test_label=results["test_label"]
        )


if __name__ == "__main__":
    main()
