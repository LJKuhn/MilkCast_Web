def C_precio_internacional(model_E):
    import streamlit as st
    import numpy as np
    import pandas as pd
    
    st.header("🌍 Predicción de Precio Internacional")
    
    # Descripción del modelo
    st.write("Predice el precio internacional de lácteos (LPE GDT) basándose en índices FAO, tipo de cambio, exportaciones y stock.")
    
    # Información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        st.write("**Tipo de Modelo:** RandomForest")
        st.write("**Preprocesamiento:** Estandarización (StandardScaler)")
        st.write("**R² Esperado:** 0.807955")
        st.write("**Variables utilizadas:**")
        st.write("- Índice de Precios de los Lácteos FAO")
        st.write("- DOLAR OFICIAL $/US$") 
        st.write("- EXPORTACIONES toneladas/mes")
        st.write("- EXISTENCIAS TOTAL")
    
    # Crear columnas para inputs
    col1, col2 = st.columns(2)
    
    with col1:
        indice_fao = st.number_input(
            label="📊 Índice de Precios de los Lácteos FAO",
            min_value=0.0,
            step=0.1,
            format="%.6f",
            key="input_indice_fao",
            help="Índice FAO de precios internacionales de lácteos"
        )
        
        dolar_oficial = st.number_input(
            label="💵 DOLAR OFICIAL $/US$ (ARS)",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_dolar_internacional",
            help="Tipo de cambio oficial USD/ARS"
        )
    
    with col2:
        exportaciones = st.number_input(
            label="🚢 EXPORTACIONES toneladas/mes",
            min_value=0.0,
            step=1.0,
            format="%.6f",
            key="input_exportaciones",
            help="Volumen mensual de exportaciones lácteas"
        )
        
        existencias_total = st.number_input(
            label="📦 EXISTENCIAS TOTAL",
            min_value=0.0,
            step=1.0,
            format="%.6f",
            key="input_existencias",
            help="Stock total de productos lácteos"
        )
    
    # Función para hacer predicción con manejo correcto del scaler
    def hacer_prediccion_precio_internacional(modelo_completo, valores):
        # Crear DataFrame con nombres de columnas correctos
        feature_names = ['Indice de Precios de los Lácteos FAO', 'DOLAR OFICIAL $/US$', 
                        'EXPORTACIONES toneladas/mes', 'EXISTENCIAS TOTAL']
        df_entrada = pd.DataFrame([valores], columns=feature_names)
        
        # Aplicar preprocesamiento si es necesario
        if modelo_completo.get('scaler') is not None:
            st.info("✅ Aplicando estandarización StandardScaler")
            datos_procesados = modelo_completo['scaler'].transform(df_entrada)
            # Convertir de vuelta a DataFrame para mantener nombres de columnas
            datos_procesados = pd.DataFrame(datos_procesados, columns=feature_names)
        else:
            st.info("✅ Usando datos originales (sin preprocesamiento - no se estandarizan ni se normalizan)")
            datos_procesados = df_entrada
        
        # Hacer predicción
        prediccion = modelo_completo['modelo'].predict(datos_procesados)
        return prediccion[0]
    
    # Botón para realizar predicción
    if st.button("🔮 Predecir Precio Internacional", key="boton_precio_internacional"):
        # Validar que no haya valores negativos
        valores = [indice_fao, dolar_oficial, exportaciones, existencias_total]
        
        if any(v < 0 for v in valores):
            st.error("❌ Por favor, ingrese solo valores positivos.")
        elif any(v == 0 for v in valores):
            st.warning("⚠️ Algunos valores son cero. Verifique que sean correctos.")
        else:
            try:
                # Verificar si el modelo es el formato completo PKL o solo el modelo sklearn
                if isinstance(model_E, dict) and 'modelo' in model_E:
                    # Es el formato completo PKL
                    prediccion = hacer_prediccion_precio_internacional(model_E, valores)
                    st.success(f"🌍 **Precio Internacional LPE GDT: USD {prediccion:.2f}/ton**")
                    
                    # Mostrar detalles adicionales
                    st.subheader("📊 Detalles de la Predicción")
                    col_det1, col_det2 = st.columns(2)
                    
                    with col_det1:
                        st.metric("LPE GDT", f"USD {prediccion:.2f}/ton")
                        if dolar_oficial > 0:
                            precio_ars = prediccion * dolar_oficial
                            st.metric("Equivalente en ARS", f"${precio_ars:,.0f}/ton")
                        st.write(f"**Tipo de Modelo:** {model_E.get('tipo_modelo', 'N/A')}")
                        st.write(f"**Preprocesamiento:** {model_E.get('preprocesamiento', 'N/A')}")
                    
                    with col_det2:
                        if 'metricas' in model_E:
                            st.write(f"**R² Esperado:** {model_E['metricas'].get('r2', 'N/A'):.6f}")
                            st.write(f"**MSE Esperado:** {model_E['metricas'].get('mse', 'N/A'):.6f}")
                    
                    # Análisis del contexto internacional
                    st.subheader("🔍 Análisis del Mercado Internacional")
                    col_analisis1, col_analisis2 = st.columns(2)
                    
                    with col_analisis1:
                        if indice_fao > 120:
                            st.warning("📈 Índice FAO Alto: Precios internacionales elevados")
                        elif indice_fao > 100:
                            st.info("📊 Índice FAO Normal: Mercado estable")
                        else:
                            st.success("📉 Índice FAO Bajo: Precios internacionales competitivos")
                    
                    with col_analisis2:
                        if exportaciones > 50000:
                            st.success("🚢 Exportaciones Altas: Buena competitividad")
                        elif exportaciones > 20000:
                            st.info("📦 Exportaciones Moderadas: Mercado normal")
                        else:
                            st.warning("📉 Exportaciones Bajas: Revisar competitividad")
                    
                else:
                    # Es solo el modelo sklearn (formato antiguo)
                    st.warning("⚠️ Modelo en formato legacy. Aplicando predicción directa.")
                    datos_entrada = np.array([valores])
                    prediccion = model_E.predict(datos_entrada)
                    st.success(f"🌍 **Precio Internacional LPE GDT: USD {prediccion[0]:.2f}/ton**")
                
                # Interpretación de resultados
                st.subheader("📈 Interpretación del Precio")
                if prediccion > 4000:
                    st.error("🔴 Precio Internacional Alto: Mercado global tensionado")
                elif prediccion > 2500:
                    st.warning("🟡 Precio Internacional Moderado: Condiciones normales")
                elif prediccion > 1500:
                    st.success("🟢 Precio Internacional Competitivo: Oportunidades de exportación")
                else:
                    st.info("🔵 Precio Internacional Bajo: Mercado deprimido")
                
                # Factores de influencia
                st.subheader("🎯 Factores Clave")
                st.write("**Índice FAO:** Refleja tendencias globales de precios lácteos")
                st.write("**Dólar Oficial:** Impacta en la competitividad exportadora")
                st.write("**Exportaciones:** Indican demanda y capacidad competitiva")
                st.write("**Existencias:** Nivel de stock afecta precios futuros")
                
            except Exception as e:
                st.error(f"❌ Error en la predicción: {str(e)}")
                
                # Información de debug
                with st.expander("🔧 Información de Debug"):
                    st.write("**Valores ingresados:**", valores)
                    st.write("**Tipo de modelo:**", type(model_E))
                    st.write("**Detalles del error:**", str(e))
    
    # Información contextual
    st.subheader("📋 Información del Mercado Internacional")
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.info("""
        **LPE GDT (Leche en Polvo Entera):**
        - Producto de exportación principal
        - Cotiza en Global Dairy Trade
        - Precio de referencia internacional
        - Base para contratos de exportación
        """)
    
    with col_info2:
        st.info("""
        **Factores Globales:**
        - Producción mundial de leche
        - Demanda de países importadores
        - Políticas comerciales
        - Condiciones climáticas globales
        """)
    
    # Mostrar información adicional
    st.subheader("💡 Consejos de Uso")
    st.info("""
    **Para obtener mejores predicciones:**
    - Use el índice FAO más reciente disponible
    - Considere el tipo de cambio al momento de la predicción
    - Las exportaciones deben ser datos mensuales consistentes
    - Las existencias reflejan la situación de stock actual
    - Monitoree tendencias trimestrales para mejor contexto
    """)