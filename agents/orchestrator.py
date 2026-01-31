"""Agent orchestration module for deep agents architecture."""

from __future__ import annotations

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from core.calculations import compute_factors, hotelling_t2, compare_model_parameters
from data.processor import read_uploaded_file, extract_time_points_and_units, validate_time_consistency
from reporting.visualizer import make_profile_plot, fig_to_png_bytes
from reporting.generator import build_pdf_report, build_excel_report, create_metadata_dict


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentResult:
    """Result from an agent execution."""
    agent_id: str
    status: AgentStatus
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


class BaseAgent:
    """Base class for all agents in the dissolution comparison system."""
    
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.status = AgentStatus.IDLE
        self.logger = logging.getLogger(f"agent.{agent_id}")
    
    async def execute(self, **kwargs) -> AgentResult:
        """Execute the agent's task.
        
        Args:
            **kwargs: Task-specific parameters
            
        Returns:
            AgentResult with execution outcome
        """
        self.status = AgentStatus.RUNNING
        start_time = asyncio.get_event_loop().time()
        
        try:
            result_data = await self._process(**kwargs)
            self.status = AgentStatus.COMPLETED
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=result_data,
                execution_time=execution_time
            )
        except Exception as e:
            self.status = AgentStatus.FAILED
            execution_time = asyncio.get_event_loop().time() - start_time
            self.logger.error(f"Agent {self.agent_id} failed: {str(e)}")
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _process(self, **kwargs) -> Any:
        """Override this method in subclasses to implement agent logic."""
        raise NotImplementedError("Subclasses must implement _process method")


class DataValidationAgent(BaseAgent):
    """Agent responsible for validating input data."""
    
    def __init__(self):
        super().__init__("data_validator", "Data Validation Agent")
    
    async def _process(self, ref_file, test_file, **kwargs) -> Dict[str, Any]:
        """Validate uploaded files and extract data."""
        # Read files
        df_ref_raw = read_uploaded_file(ref_file)
        df_test_raw = read_uploaded_file(test_file)
        
        # Extract time points and units
        time_ref, ref_units, df_ref_num = extract_time_points_and_units(df_ref_raw)
        time_test, test_units, df_test_num = extract_time_points_and_units(df_test_raw)
        
        # Validate time consistency
        if not validate_time_consistency(time_ref, time_test):
            raise ValueError("Time points are not consistent between reference and test files")
        
        return {
            "time_points": time_ref,
            "ref_units": ref_units,
            "test_units": test_units,
            "df_ref_num": df_ref_num,
            "df_test_num": df_test_num,
            "n_units": ref_units.shape[0]
        }


class F1F2Agent(BaseAgent):
    """Agent for f1/f2 similarity factor calculation."""
    
    def __init__(self):
        super().__init__("f1_f2_calculator", "F1/F2 Calculation Agent")
    
    async def _process(self, ref_mean, test_mean, **kwargs) -> Dict[str, Any]:
        """Calculate f1 and f2 factors."""
        f1, f2 = compute_factors(ref_mean, test_mean)
        
        return {
            "f1": f1,
            "f2": f2,
            "similar": (f1 <= 15.0) and (f2 >= 50.0),
            "conclusion": f"{'SIMILARES' if (f1 <= 15.0 and f2 >= 50.0) else 'NO SIMILARES'} (criterio típico: f1 ≤ 15 y f2 ≥ 50)."
        }


class MultivariateAgent(BaseAgent):
    """Agent for multivariate Hotelling T² analysis."""
    
    def __init__(self):
        super().__init__("multivariate_analyzer", "Multivariate Analysis Agent")
    
    async def _process(self, ref_units, test_units, **kwargs) -> Dict[str, Any]:
        """Perform Hotelling T² analysis."""
        T2, df1, df2, ci90 = hotelling_t2(ref_units, test_units)
        
        return {
            "T2": T2,
            "df1": df1,
            "df2": df2,
            "ci90": ci90,
            "similar": T2 <= ci90,
            "conclusion": f"{'SIMILARES' if T2 <= ci90 else 'NO SIMILARES'} (T²={T2:.4f} vs Límite 90%={ci90:.4f})."
        }


class ModelDependentAgent(BaseAgent):
    """Agent for model-dependent analysis."""
    
    def __init__(self):
        super().__init__("model_dependent_analyzer", "Model-Dependent Analysis Agent")
    
    async def _process(self, ref_units, test_units, time_points, model_name="weibull", **kwargs) -> Dict[str, Any]:
        """Perform model-dependent analysis."""
        result = compare_model_parameters(ref_units, test_units, time_points, model_name)
        
        return {
            "model_name": model_name,
            "ref_params": result["ref_params"],
            "test_params": result["test_params"],
            "T2": result["T2"],
            "df1": result["df1"],
            "df2": result["df2"],
            "ci90": result["ci90"],
            "similar": result["similar"],
            "conclusion": f"{'SIMILARES' if result['similar'] else 'NO SIMILARES'} (T²(parámetros)={result['T2']:.4f} vs Límite 90%={result['ci90']:.4f})."
        }


class VisualizationAgent(BaseAgent):
    """Agent for generating visualizations."""
    
    def __init__(self):
        super().__init__("visualizer", "Visualization Agent")
    
    async def _process(self, time_points, ref_mean, test_mean, ref_sd, test_sd, 
                      ref_label, test_label, title, **kwargs) -> Dict[str, Any]:
        """Generate profile plot."""
        fig = make_profile_plot(
            time_points, ref_mean, test_mean, ref_label, test_label, title, ref_sd, test_sd
        )
        
        return {
            "profile_plot_bytes": fig_to_png_bytes(fig),
            "figure": fig
        }


class ReportGenerationAgent(BaseAgent):
    """Agent for generating reports."""
    
    def __init__(self):
        super().__init__("report_generator", "Report Generation Agent")
    
    async def _process(self, metadata, method_name, conclusion, tables, figures, **kwargs) -> Dict[str, Any]:
        """Generate PDF and Excel reports."""
        pdf_bytes = build_pdf_report(
            title="Reporte – Comparación de perfiles de disolución",
            metadata=metadata,
            method_name=method_name,
            conclusion=conclusion,
            tables=tables,
            figures=figures,
        )
        
        xlsx_bytes = build_excel_report(tables)
        
        return {
            "pdf_bytes": pdf_bytes,
            "xlsx_bytes": xlsx_bytes
        }


class AgentOrchestrator:
    """Orchestrates multiple agents for dissolution comparison analysis."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.results: Dict[str, AgentResult] = {}
        self.logger = logging.getLogger("orchestrator")
        
        # Initialize agents
        self._register_agents()
    
    def _register_agents(self):
        """Register all available agents."""
        self.agents = {
            "data_validator": DataValidationAgent(),
            "f1_f2_calculator": F1F2Agent(),
            "multivariate_analyzer": MultivariateAgent(),
            "model_dependent_analyzer": ModelDependentAgent(),
            "visualizer": VisualizationAgent(),
            "report_generator": ReportGenerationAgent(),
        }
    
    async def execute_analysis(
        self, 
        method: str,
        ref_file,
        test_file,
        metadata: Dict[str, str],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute complete dissolution comparison analysis using agents.
        
        Args:
            method: Analysis method ("f1_f2", "multivariate", "model_dependent")
            ref_file: Reference file
            test_file: Test file
            metadata: Report metadata
            **kwargs: Additional parameters
            
        Returns:
            Complete analysis results
        """
        # Step 1: Data validation
        self.logger.info("Starting data validation...")
        data_result = await self.agents["data_validator"].execute(ref_file=ref_file, test_file=test_file)
        
        if data_result.status == AgentStatus.FAILED:
            raise Exception(f"Data validation failed: {data_result.error}")
        
        self.results["data_validation"] = data_result
        data = data_result.data
        
        # Calculate summary statistics
        ref_mean = data["ref_units"].mean(axis=0)
        test_mean = data["test_units"].mean(axis=0)
        ref_sd = data["ref_units"].std(axis=0, ddof=1)
        test_sd = data["test_units"].std(axis=0, ddof=1)
        
        # Step 2: Generate visualization
        self.logger.info("Generating visualization...")
        viz_result = await self.agents["visualizer"].execute(
            time_points=data["time_points"],
            ref_mean=ref_mean,
            test_mean=test_mean,
            ref_sd=ref_sd,
            test_sd=test_sd,
            ref_label=f"Referencia (Lote {metadata.get('Lote referencia', 'N/A')})",
            test_label=f"Test (Lote {metadata.get('Lote test', 'N/A')})",
            title="Perfil de disolución",
            **kwargs
        )
        
        if viz_result.status == AgentStatus.FAILED:
            self.logger.warning(f"Visualization failed: {viz_result.error}")
        
        self.results["visualization"] = viz_result
        
        # Step 3: Method-specific analysis
        analysis_result = None
        if method == "f1_f2":
            self.logger.info("Running f1/f2 analysis...")
            analysis_result = await self.agents["f1_f2_calculator"].execute(
                ref_mean=ref_mean, test_mean=test_mean
            )
        elif method == "multivariate":
            self.logger.info("Running multivariate analysis...")
            analysis_result = await self.agents["multivariate_analyzer"].execute(
                ref_units=data["ref_units"], test_units=data["test_units"]
            )
        elif method == "model_dependent":
            self.logger.info("Running model-dependent analysis...")
            model_name = kwargs.get("model_name", "weibull")
            analysis_result = await self.agents["model_dependent_analyzer"].execute(
                ref_units=data["ref_units"],
                test_units=data["test_units"],
                time_points=data["time_points"],
                model_name=model_name
            )
        
        if analysis_result and analysis_result.status == AgentStatus.FAILED:
            raise Exception(f"Analysis failed: {analysis_result.error}")
        
        self.results["analysis"] = analysis_result
        
        # Step 4: Prepare report data
        tables = self._prepare_report_tables(data, analysis_result.data if analysis_result else None)
        figures = {"Perfil de disolución (media ± DE)": viz_result.data["profile_plot_bytes"]} if viz_result.status == AgentStatus.COMPLETED else {}
        
        # Step 5: Generate reports
        self.logger.info("Generating reports...")
        report_result = await self.agents["report_generator"].execute(
            metadata=metadata,
            method_name=method,
            conclusion=analysis_result.data["conclusion"] if analysis_result else "No conclusion",
            tables=tables,
            figures=figures
        )
        
        if report_result.status == AgentStatus.FAILED:
            self.logger.warning(f"Report generation failed: {report_result.error}")
        
        self.results["report"] = report_result
        
        # Compile final results
        return {
            "data": data,
            "analysis": analysis_result.data if analysis_result else None,
            "visualization": viz_result.data if viz_result.status == AgentStatus.COMPLETED else None,
            "reports": report_result.data if report_result.status == AgentStatus.COMPLETED else None,
            "execution_summary": self._get_execution_summary()
        }
    
    def _prepare_report_tables(self, data: Dict, analysis_data: Dict) -> Dict[str, pd.DataFrame]:
        """Prepare tables for report generation."""
        import pandas as pd
        
        tables = {}
        
        # Summary statistics table
        from data.processor import calculate_summary_stats
        overview = calculate_summary_stats(
            data["ref_units"], data["test_units"], data["time_points"],
            "Ref", "Test"
        )
        tables["Resumen por tiempo"] = overview
        
        # Method-specific tables
        if analysis_data:
            if "f1" in analysis_data:  # f1/f2 method
                tables["Cálculo f1/f2"] = pd.DataFrame({
                    "Métrica": ["f1", "f2"],
                    "Valor": [round(analysis_data["f1"], 4), round(analysis_data["f2"], 4)],
                    "Criterio": ["≤ 15", "≥ 50"],
                    "Cumple": [analysis_data["f1"] <= 15, analysis_data["f2"] >= 50],
                })
            elif "T2" in analysis_data:  # Multivariate or model-dependent
                if "ref_params" in analysis_data:  # Model-dependent
                    ref_params = pd.DataFrame(analysis_data["ref_params"], columns=["a", "b", "c"])
                    test_params = pd.DataFrame(analysis_data["test_params"], columns=["a", "b", "c"])
                    ref_params.insert(0, "Unidad", np.arange(1, len(ref_params) + 1))
                    test_params.insert(0, "Unidad", np.arange(1, len(test_params) + 1))
                    
                    model_name = analysis_data.get("model_name", "Unknown")
                    tables[f"Parámetros modelo (Referencia) - {model_name}"] = ref_params.round(6)
                    tables[f"Parámetros modelo (Prueba) - {model_name}"] = test_params.round(6)
                    tables["Comparación en parámetros"] = pd.DataFrame({
                        "Métrica": ["Hotelling T² (parámetros)", "Límite crítico (90%)", "df1", "df2"],
                        "Valor": [round(analysis_data["T2"], 6), round(analysis_data["ci90"], 6), 
                                analysis_data["df1"], analysis_data["df2"]],
                    })
                else:  # Multivariate
                    tables["Comparación multivariante"] = pd.DataFrame({
                        "Métrica": ["Hotelling T²", "Límite crítico (90%)", "df1", "df2"],
                        "Valor": [round(analysis_data["T2"], 6), round(analysis_data["ci90"], 6),
                                analysis_data["df1"], analysis_data["df2"]],
                    })
        
        return tables
    
    def _get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of agent executions."""
        summary = {}
        for agent_id, result in self.results.items():
            summary[agent_id] = {
                "status": result.status.value,
                "execution_time": result.execution_time,
                "error": result.error
            }
        return summary
    
    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """Get status of a specific agent."""
        if agent_id in self.agents:
            return self.agents[agent_id].status
        return None
    
    def reset(self):
        """Reset all agents and results."""
        for agent in self.agents.values():
            agent.status = AgentStatus.IDLE
        self.results.clear()
