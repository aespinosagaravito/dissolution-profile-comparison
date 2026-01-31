# Comparación de Perfiles de Disolución - Arquitectura Modular

Esta aplicación ha sido reestructurada en una arquitectura modular diseñada para soportar **Deep Agents** y facilitar el mantenimiento, escalabilidad y testing.

## 📁 Estructura del Proyecto

```
COMPARACION DISOLUCION/
├── main.py                          # Punto de entrada principal
├── requirements.txt                 # Dependencias del proyecto
├── config/                          # Configuración y constantes
│   ├── __init__.py
│   └── constants.py                # Constantes, configuraciones, mensajes
├── core/                           # Motor de cálculos matemáticos
│   ├── __init__.py
│   ├── calculations.py            # Cálculos f1/f2, Hotelling T²
│   └── models.py                   # Modelos matemáticos (Weibull, Logístico, Lineal)
├── data/                           # Procesamiento de datos
│   ├── __init__.py
│   └── processor.py                # Lectura de archivos, validación, parsing
├── reporting/                      # Generación de reportes
│   ├── __init__.py
│   ├── generator.py                # PDF, Excel report generation
│   └── visualizer.py               # Gráficos y visualizaciones
├── ui/                             # Interfaz de usuario
│   ├── __init__.py
│   └── streamlit_app.py            # Componentes UI de Streamlit
├── agents/                         # Arquitectura de Deep Agents
│   ├── __init__.py
│   └── orchestrator.py              # Orquestación de agentes
└── versiones/                      # Versiones anteriores (preservadas)
```

## 🏗️ Arquitectura de Módulos

### 1. **config/constants.py**
- Centraliza todas las constantes y configuraciones
- Umbrales de decisión (f1 ≤ 15, f2 ≥ 50)
- Colores y estilos de gráficos
- Mensajes de error y éxito
- Plantillas de ayuda para métodos

### 2. **core/calculations.py**
- Motor matemático principal
- `compute_factors()`: Cálculo de factores f1/f2
- `hotelling_t2()`: Análisis multivariante
- `compare_model_parameters()`: Comparación dependiente de modelo
- Funciones de evaluación de similitud

### 3. **core/models.py**
- Modelos matemáticos para ajuste de curvas
- `logistic_model()`: Modelo logístico de 3 parámetros
- `weibull_model()`: Modelo Weibull
- `linear_model()`: Modelo lineal con saturación
- Registro de modelos y gestión de parámetros iniciales

### 4. **data/processor.py**
- Procesamiento robusto de datos
- Lectura de archivos Excel/CSV
- Extracción automática de puntos de tiempo
- Validación de consistencia de datos
- Cálculo de estadísticas resumen

### 5. **reporting/generator.py**
- Generación de reportes profesionales
- Creación de PDFs con tablas y gráficos
- Exportación a Excel con múltiples hojas
- Gestión de metadatos y nombres de archivo

### 6. **reporting/visualizer.py**
- Visualizaciones de alta calidad
- Gráficos de perfiles de disolución
- Gráficos de ajuste de modelos
- Gráficos de residuos para diagnóstico
- Configuración de estilos consistentes

### 7. **ui/streamlit_app.py**
- Componentes modulares de UI
- Renderizado de resultados por método
- Gestión de descargas
- Ayuda interactiva y explicaciones

## 🤖 Arquitectura de Deep Agents

### Agentes Especializados

1. **DataValidationAgent**
   - Validación de archivos de entrada
   - Extracción y validación de datos
   - Verificación de consistencia temporal

2. **F1F2Agent**
   - Cálculo especializado de factores f1/f2
   - Evaluación de similitud según criterios FDA

3. **MultivariateAgent**
   - Análisis multivariante Hotelling T²
   - Manejo de matrices de covarianza

4. **ModelDependentAgent**
   - Ajuste de modelos por unidad
   - Comparación en espacio de parámetros

5. **VisualizationAgent**
   - Generación de visualizaciones
   - Optimización de gráficos para reportes

6. **ReportGenerationAgent**
   - Compilación de reportes PDF/Excel
   - Gestión de metadatos y formato

### Orquestador

El `AgentOrchestrator` coordina la ejecución de agentes:
- Ejecución asíncrona de tareas
- Manejo de errores y recuperación
- Registro de tiempos de ejecución
- Composición de resultados finales

## 🚀 Características Principales

### ✅ Modularidad
- Cada componente tiene una responsabilidad única
- Interfaces claras entre módulos
- Fácil testing unitario

### ✅ Escalabilidad
- Arquitectura de agentes permite paralelización
- Fácil adición de nuevos métodos de análisis
- Componentes reutilizables

### ✅ Mantenimiento
- Separación de concerns clara
- Configuración centralizada
- Código documentado

### ✅ Robustez
- Validación exhaustiva de datos
- Manejo de errores granular
- Logging integrado

## 📦 Instalación y Ejecución

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecutar aplicación tradicional
```bash
streamlit run main.py
```

### 3. Usar arquitectura de agentes (experimental)
- En la interfaz, activar "Usar arquitectura de agentes"
- Los agentes ejecutarán el análisis de forma asíncrona

## 🔧 Modo de Uso

### Interfaz Streamlit
1. **Cargar archivos**: Referencia (pre-cambio) y Prueba (post-cambio)
2. **Seleccionar método**: f1/f2, Multivariante, o Dependiente de modelo
3. **Ingresar metadatos**: Lotes, producto, analista, etc.
4. **Ejecutar análisis**: Tradicional o con agentes
5. **Descargar reportes**: PDF completo y Excel con tablas

### Opciones Avanzadas
- **Mostrar detalles**: Tablas numéricas completas
- **Explicación del método**: Fórmulas y criterios
- **Modelo dependiente**: Selección de Weibull/Logístico/Lineal

## 🧪 Testing

La arquitectura modular facilita el testing:

```python
# Test de cálculos f1/f2
from core.calculations import compute_factors
f1, f2 = compute_factors(ref_mean, test_mean)

# Test de procesamiento de datos
from data.processor import extract_time_points_and_units
times, units, df = extract_time_points_and_units(dataframe)

# Test de agentes
from agents.orchestrator import AgentOrchestrator
orchestrator = AgentOrchestrator()
result = await orchestrator.execute_analysis(...)
```

## 🔄 Comparación con Versión Original

| Característica | Original | Nueva Arquitectura |
|---------------|----------|-------------------|
| **Estructura** | 1 archivo monolítico | 9 módulos especializados |
| **Testing** | Difícil (acoplamiento) | Fácil (modular) |
| **Escalabilidad** | Limitada | Alta (agentes) |
| **Mantenimiento** | Complejo | Simplificado |
| **Extensibilidad** | Baja | Alta |
| **Performance** | Síncrona | Asíncrona (opcional) |

## 📈 Beneficios de la Arquitectura de Agents

1. **Paralelización**: Agentes pueden ejecutarse concurrentemente
2. **Resiliencia**: Fallos en un agente no afectan a otros
3. **Monitoring**: Tiempos de ejecución por componente
4. **Reutilización**: Agentes pueden usarse en otros contextos
5. **Testing**: Cada agente puede probarse independientemente

## 🔮 Futuras Extensiones

- **Agentes de ML**: Para predicción y clasificación
- **Agentes de Validación**: Para verificación regulatoria
- **Agentes de Optimización**: Para diseño experimental
- **API REST**: Para integración con otros sistemas
- **Base de Datos**: Para almacenamiento histórico

---

**Nota**: La aplicación mantiene compatibilidad total con la versión original mientras proporciona una base sólida para desarrollo futuro.
