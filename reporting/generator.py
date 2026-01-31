"""Report generation module for PDF and Excel outputs."""

from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Dict, List

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config.constants import (
    PDF_MARGIN, PDF_IMAGE_HEIGHT, PDF_IMAGE_WIDTH, 
    MAX_TABLE_ROWS, DEFAULT_MEDIUM
)


def df_to_rl_table(df: pd.DataFrame, max_rows: int = MAX_TABLE_ROWS) -> Table:
    """Convert a DataFrame into a ReportLab Table.
    
    Args:
        df: DataFrame to convert
        max_rows: Maximum number of rows to display
        
    Returns:
        ReportLab Table object
    """
    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = pd.concat([df2.head(max_rows), pd.DataFrame([{"…": "…"}])], ignore_index=True)

    data = [list(df2.columns)] + df2.astype(str).values.tolist()
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ])
    )
    return tbl


def build_pdf_report(
    title: str,
    metadata: Dict[str, str],
    method_name: str,
    conclusion: str,
    tables: Dict[str, pd.DataFrame],
    figures: Dict[str, bytes],
) -> bytes:
    """Create a PDF report as bytes.
    
    Args:
        title: Report title
        metadata: Dictionary with metadata information
        method_name: Name of the method used
        conclusion: Similarity conclusion
        tables: Dictionary of DataFrames to include
        figures: Dictionary of figure images (bytes) to include
        
    Returns:
        PDF report as bytes
    """
    styles = getSampleStyleSheet()
    story: List[object] = []

    # Title
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.3 * cm))

    # Metadata
    meta_lines = "<br/>".join([f"<b>{k}:</b> {v}" for k, v in metadata.items()])
    story.append(Paragraph(meta_lines, styles["BodyText"]))
    story.append(Spacer(1, 0.4 * cm))

    # Method and conclusion
    story.append(Paragraph(f"<b>Método:</b> {method_name}", styles["Heading2"]))
    story.append(Paragraph(f"<b>Conclusión:</b> {conclusion}", styles["BodyText"]))
    story.append(Spacer(1, 0.4 * cm))

    # Tables
    for name, df in tables.items():
        story.append(Paragraph(name, styles["Heading3"]))
        story.append(df_to_rl_table(df))
        story.append(Spacer(1, 0.35 * cm))

    # Figures
    for name, png_bytes in figures.items():
        story.append(Paragraph(name, styles["Heading3"]))
        img = RLImage(io.BytesIO(png_bytes))
        img.drawHeight = PDF_IMAGE_HEIGHT * cm
        img.drawWidth = PDF_IMAGE_WIDTH * cm
        story.append(img)
        story.append(Spacer(1, 0.35 * cm))

    # Build PDF
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=PDF_MARGIN * cm,
        leftMargin=PDF_MARGIN * cm,
        topMargin=PDF_MARGIN * cm,
        bottomMargin=PDF_MARGIN * cm,
    )
    doc.build(story)
    buf.seek(0)
    return buf.read()


def build_excel_report(tables: Dict[str, pd.DataFrame]) -> bytes:
    """Create an Excel report with multiple sheets.
    
    Args:
        tables: Dictionary of DataFrames to include as sheets
        
    Returns:
        Excel file as bytes
    """
    if not tables:
        # Create a default empty table if no tables provided
        tables = {"Sin datos": pd.DataFrame({"Mensaje": ["No hay datos disponibles para exportar"]})}
    
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, df in tables.items():
            # Sanitize sheet name (Excel limits)
            safe = re.sub(r"[^A-Za-z0-9_ ]+", "", sheet_name)[:31].strip() or "Sheet"
            df.to_excel(writer, index=False, sheet_name=safe)
    buf.seek(0)
    return buf.read()


def create_metadata_dict(
    product_name: str,
    active_name: str,
    reference_lot: str,
    test_lot: str,
    medium: str = DEFAULT_MEDIUM,
    apparatus: str = "",
    analyst: str = "",
    notes: str = "",
    n_units: int = 0,
    time_points: List[float] = None,
) -> Dict[str, str]:
    """Create metadata dictionary for reports.
    
    Args:
        product_name: Product name/code
        active_name: Active ingredient/strength
        reference_lot: Reference lot number
        test_lot: Test lot number
        medium: Dissolution medium
        apparatus: Apparatus type and RPM
        analyst: Analyst name
        notes: Additional notes
        n_units: Number of units per batch
        time_points: List of time points
        
    Returns:
        Metadata dictionary
    """
    if time_points is None:
        time_points = []
    
    return {
        "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Producto": product_name or "(no indicado)",
        "Activo": active_name or "(no indicado)",
        "Lote referencia": reference_lot,
        "Lote test": test_lot,
        "Medio": medium or "(no indicado)",
        "Aparato/RPM": apparatus or "(no indicado)",
        "Analista": analyst or "(no indicado)",
        "Notas": notes or "(ninguna)",
        "N unidades": str(n_units),
        "Tiempos": ", ".join([str(int(t)) if float(t).is_integer() else str(t) for t in time_points]),
    }


def generate_filename(
    product_name: str,
    reference_lot: str,
    test_lot: str,
    report_type: str = "reporte",
    file_extension: str = "pdf"
) -> str:
    """Generate standardized filename for reports.
    
    Args:
        product_name: Product name
        reference_lot: Reference lot
        test_lot: Test lot
        report_type: Type of report (reporte, tablas, etc.)
        file_extension: File extension
        
    Returns:
        Generated filename
    """
    from data.processor import sanitize_filename_token
    
    fn_product = sanitize_filename_token(product_name, 20)
    fn_ref = sanitize_filename_token(reference_lot, 20)
    fn_test = sanitize_filename_token(test_lot, 20)
    
    return f"{report_type}_disolucion_{fn_product}_{fn_ref}_vs_{fn_test}.{file_extension}"
