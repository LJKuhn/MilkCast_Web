# 🥛 MilkCast - Predicción de Precios en el Sector Lácteo

Una aplicación web desarrollada con **Streamlit** que utiliza **Machine Learning** para predecir precios de la leche en Argentina, orientada a productores agropecuarios para la toma de decisiones informadas.

![MilkCast](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 🎯 Descripción

MilkCast es una herramienta de análisis predictivo desarrollada como parte de las Prácticas Profesionales Supervisadas. La aplicación permite a los productores agropecuarios anticiparse a distintos escenarios económicos en un contexto de cambios como la reactivación del crédito, el levantamiento del cepo cambiario y las modificaciones en las retenciones.

### ✨ Funcionalidades Principales

- **📊 Análisis Exploratorio de Datos (EDA)**: Visualización interactiva de datos del sector lácteo
- **🔮 Predicción por IPC y Dólar**: Modelo basado en Índice de Precios al Consumidor y tipo de cambio
- **🥛 Predicción por Productos**: Predicción basada en precios de leche entera, queso y yogur
- **🎯 Predicción de Rentabilidad**: Modelo A - Análisis de rentabilidad con 6 variables
- **💰 Predicción de Costos**: Modelo B - Estimación de costos de producción
- **🧀 Predicción Precio Queso**: Modelo D - Precio de queso cuartirolo
- **🌍 Predicción Internacional**: Modelo E - Precios internacionales LPE GDT
- **🐄 Predicción Novillos**: Modelo F - Precios del mercado ganadero
- **📊 Variables Macroeconómicas**: Modelo G - Predicción simple con IPC y dólar
- **🥛 Productos Lácteos Específicos**: Modelo H - Basado en precios minoristas
- **📈 Visualizaciones Dinámicas**: Gráficos interactivos con Plotly

## 🏗️ Estructura del Proyecto

```
MilkCast_Web/
├── 📄 main.py                           # Aplicación principal Streamlit
├── 📄 requirements.txt                  # Dependencias de Python
├── 📄 README.md                         # Este archivo
│
├── 📁 componentes/                      # Módulos de la aplicación
│   ├── __init__.py                      # Inicializador del paquete
│   ├── componente_eda.py                # Análisis exploratorio y visualizaciones
│   ├── componente_prediccion.py         # Predicción con IPC y dólar
│   └── componente_clasificacion.py      # Predicción con productos lácteos
│
├── 📁 modelos/                          # Modelos ML entrenados
│   ├── modelo_regresion-Precio-IPC-Dolar.pkl           # Modelo IPC-Dólar (original)
│   ├── modelo_regresion-Precio-ComEnt-Queso-Yogur.pkl  # Modelo productos (original)
│   ├── modelo_A_rentabilidad.pkl                       # Modelo A - Rentabilidad
│   ├── modelo_B_costos.pkl                             # Modelo B - Costos
│   ├── modelo_D_precio_queso.pkl                       # Modelo D - Precio Queso
│   ├── modelo_E_precio_internacional.pkl               # Modelo E - Internacional
│   ├── modelo_F_precio_novillos.pkl                    # Modelo F - Novillos
│   ├── modelo_G_variables_macroeconomicas.pkl          # Modelo G - Variables Macro
│   ├── modelo_H_productos_lacteos.pkl                  # Modelo H - Productos Lácteos
│   └── archivo.csv                      # Datos de referencia
│
├── 📁 Imagenes/                         # Recursos gráficos
│   ├── grafico.png                      # Gráfico de ejemplo
│   └── mapa.png                         # Mapa de unidades productivas
│
└── 📁 .devcontainer/                    # Configuración para desarrollo
    └── devcontainer.json                # Configuración Docker container
```

## 🚀 Instalación y Ejecución

### Prerrequisitos
- Python 3.11+ (recomendado)
- pip (gestor de paquetes de Python)

### Instalación Local

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/LJKuhn/MilkCast_Web.git
   cd MilkCast_Web
   ```

2. **Crear entorno virtual** (recomendado)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar la aplicación**
   ```bash
   streamlit run main.py
   ```

5. **Abrir en el navegador**
   - La aplicación se ejecutará en `http://localhost:8501`

### 🐳 Desarrollo con DevContainer

Este proyecto incluye configuración para **Visual Studio Code DevContainers**:

1. Abrir el proyecto en VS Code
2. Instalar la extensión "Dev Containers"
3. Presionar `Ctrl+Shift+P` y seleccionar "Dev Containers: Reopen in Container"
4. El container se configurará automáticamente con Python 3.11 y todas las dependencias

## 🔧 Tecnologías y Dependencias

### Framework Principal
- **Streamlit**: Framework para aplicaciones web de datos
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Computación científica

### Machine Learning
- **Scikit-learn**: Modelos de regresión
- **Pickle**: Serialización de modelos pre-entrenados

### Visualización
- **Plotly Express**: Gráficos interactivos
- **Matplotlib**: Visualizaciones estáticas
- **Seaborn**: Gráficos estadísticos

### Utilidades
- **gdown**: Descarga automática de modelos desde Google Drive
- **SciPy**: Herramientas científicas (interpolación, filtros)

## 📊 Modelos de Machine Learning

### 🏦 Modelo IPC-Dólar
- **Archivo**: `modelo_regresion-Precio-IPC-Dolar.pkl`
- **Variables predictoras**: 
  - IPC (Índice de Precios al Consumidor)
  - Valor del dólar
- **Objetivo**: Predicción del precio del litro de leche
- **Uso**: Escenarios macroeconómicos

### 🥛 Modelo Productos Lácteos
- **Archivo**: `modelo_regresion-Precio-ComEnt-Queso-Yogur.pkl`
- **Variables predictoras**:
  - Precio leche común entera ($/litro)
  - Precio queso tipo cuartirolo ($/kg)
  - Precio yogur
- **Objetivo**: Predicción basada en productos del mercado
- **Uso**: Análisis de productos específicos

## 🖥️ Interfaz de Usuario

La aplicación cuenta con **10 pestañas principales**:

### 1. 📊 "Datos y Gráficos"
- Análisis exploratorio interactivo
- Visualizaciones de tendencias históricas
- Mapas de unidades productivas
- Información contextual del sector

### 2. 💱 "IPC y Dólar"
- Input para valores de IPC y tipo de cambio
- Instrucciones detalladas para cálculo de IPC acumulado
- Predicción en tiempo real
- Visualización de resultados

### 3. � "Productos Básicos"
- Input para precios de productos lácteos
- Predicción basada en precios del mercado
- Comparación entre productos
- Análisis de correlaciones

### 4. 🎯 "Rentabilidad" (Modelo A)
- Predicción de rentabilidad con 6 variables
- Análisis de costos vs ingresos
- Preprocesamiento con StandardScaler
- Interpretación de niveles de rentabilidad

### 5. 💰 "Costos" (Modelo B)
- Estimación de costos de producción
- Variables macroeconómicas y sectoriales
- Sin preprocesamiento (datos originales)
- Análisis de factores influyentes

### 6. 🧀 "Precio Queso" (Modelo D)
- Predicción de precio de queso cuartirolo
- Basado en precios lácteos e índices
- Preprocesamiento con StandardScaler
- Análisis de relaciones queso/leche

### 7. 🌍 "Internacional" (Modelo E)
- Predicción de precios internacionales LPE GDT
- Índices FAO y variables comerciales
- Preprocesamiento con StandardScaler
- Conversión automática USD/ARS

### 8. 🐄 "Novillos" (Modelo F)
- Predicción de precios de novillos
- Variables del mercado ganadero y macroeconómicas
- Preprocesamiento con StandardScaler
- Comparación con precios de vaquillonas

### 9. 📊 "Variables Macro" (Modelo G)
- Predicción simple con IPC y dólar
- Modelo económico básico (2 variables)
- Sin preprocesamiento (datos originales)
- Análisis macroeconómico

### 10. 🥛 "Productos H" (Modelo H)
- Predicción basada en precios minoristas
- Leche, queso y yogur específicos
- Sin preprocesamiento (datos originales)
- Calculadora de relaciones de precios

## 🔄 Funcionalidades Técnicas

### Descarga Automática de Modelos
- Los modelos se descargan automáticamente desde Google Drive
- Sistema de caché para evitar descargas innecesarias
- Manejo de errores en la descarga

### Procesamiento de Datos
- Normalización con MinMaxScaler
- Interpolación de datos faltantes
- Filtros de suavizado para visualizaciones

## 💡 Uso Práctico

### Para Productores:
1. **Planificación**: Utilizar predicciones para planificar producción
2. **Toma de decisiones**: Anticiparse a cambios de precios
3. **Análisis de mercado**: Comparar con datos históricos

### Para Analistas:
1. **EDA**: Explorar patrones en los datos
2. **Validación**: Comparar predicciones con datos reales
3. **Investigación**: Analizar correlaciones entre variables

## 🚀 Próximas Mejoras

- [ ] Incorporación de más variables macroeconómicas
- [ ] Modelos de Deep Learning (LSTM)
- [ ] API REST para integración externa
- [ ] Dashboard administrativo
- [ ] Alertas automáticas por email
- [ ] Exportación de reportes PDF
- [ ] Integración con bases de datos en tiempo real

## 🤝 Contribuir

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -m 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abrir un Pull Request

## 📈 Rendimiento

- ⚡ Carga rápida de modelos con pickle
- 🔄 Descarga automática de dependencias
- 📱 Interfaz responsive
- 🎨 Visualizaciones optimizadas con Plotly

## 👥 Autor

- **Desarrollador**: [LJKuhn](https://github.com/LJKuhn)(1221kuhn@gmail.com)
- **Proyecto**: Prácticas Profesionales Supervisadas 2025
- **Institución**: Universidad Nacional de Rafaela (https://www.unraf.edu.ar/)

## 📄 Licencia

Este proyecto está desarrollado con fines educativos y de investigación como parte de las Prácticas Profesionales Supervisadas.

## 📞 Soporte

Si encuentras algún problema o tienes sugerencias:

1. 🐛 [Reportar bugs](https://github.com/LJKuhn/MilkCast_Web/issues)
2. 💡 [Sugerir mejoras](https://github.com/LJKuhn/MilkCast_Web/discussions)
3. 📧 Contactar al desarrollador

---

**🚀 ¡Ejecuta MilkCast y transforma datos en decisiones para el sector lácteo argentino!** 🇦🇷