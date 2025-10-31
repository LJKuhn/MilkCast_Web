def C_variables_macro(model_G):
    import streamlit as st
    import numpy as np
    import pandas as pd
    
    st.header("📊 Predicción con Variables Macroeconómicas")
    
    # Descripción del modelo
    st.write("Predice el precio de la leche basándose únicamente en variables macroeconómicas clave: IPC y tipo de cambio oficial.")
    
    # Información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        st.write("**Tipo de Modelo:** RandomForest")
        st.write("**Preprocesamiento:** Datos Originales (sin estandarización)")
        st.write("**R² Esperado:** 0.999706")
        st.write("**Variables utilizadas:**")
        st.write("- IPC-Mensual")
        st.write("- DOLAR OFICIAL $/US$")
    
    # Crear columnas para inputs
    col1, col2 = st.columns(2)
    
    with col1:
        ipc_mensual = st.number_input(
            label="📊 IPC-Mensual",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_ipc_macro",
            help="Índice de Precios al Consumidor mensual acumulado"
        )
        
        # Información adicional sobre IPC
        with st.expander("ℹ️ Cómo calcular el IPC"):
            st.write("""
            **Para calcular el IPC acumulado:**
            - Valor base: 98 (inicio de serie)
            - Último valor oficial: 7864,1257 (enero 2025)
            - Para febrero 2025: 7864,1257 × 1,021 (2,1% inflación)
            - Para marzo 2025: resultado anterior × 1,037 (3,7% inflación)
            - Continuar multiplicando por coeficientes mensuales
            """)
    
    with col2:
        dolar_oficial = st.number_input(
            label="💵 DOLAR OFICIAL $/US$ (ARS)",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_dolar_macro",
            help="Tipo de cambio oficial USD/ARS"
        )
        
        # Información adicional sobre dólar
        with st.expander("ℹ️ Información del Tipo de Cambio"):
            st.write("""
            **Tipo de Cambio Oficial:**
            - Cotización del Banco Central (BCRA)
            - Diferente del dólar blue o financiero
            - Impacta directamente en costos de insumos
            - Afecta competitividad exportadora
            """)
    
    # Función para hacer predicción con manejo correcto del scaler
    def hacer_prediccion_variables_macro(modelo_completo, valores):
        # Crear DataFrame con nombres de columnas correctos
        feature_names = ['IPC-Mensual', 'DOLAR OFICIAL $/US$']
        df_entrada = pd.DataFrame([valores], columns=feature_names)
        
        # Aplicar preprocesamiento si es necesario
        if modelo_completo.get('scaler') is not None:
            st.info("✅ Aplicando preprocesamiento")
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
    if st.button("🔮 Predecir Precio con Variables Macro", key="boton_variables_macro"):
        # Validar que no haya valores negativos
        valores = [ipc_mensual, dolar_oficial]
        
        if any(v < 0 for v in valores):
            st.error("❌ Por favor, ingrese solo valores positivos.")
        elif any(v == 0 for v in valores):
            st.warning("⚠️ Algunos valores son cero. Verifique que sean correctos.")
        else:
            try:
                # Verificar si el modelo es el formato completo PKL o solo el modelo sklearn
                if isinstance(model_G, dict) and 'modelo' in model_G:
                    # Es el formato completo PKL
                    prediccion = hacer_prediccion_variables_macro(model_G, valores)
                    st.success(f"🥛 **Precio de Leche Predicho: ${prediccion:.6f}/litro**")
                    
                    # Mostrar detalles adicionales
                    st.subheader("📊 Detalles de la Predicción")
                    col_det1, col_det2 = st.columns(2)
                    
                    with col_det1:
                        st.metric("Precio/litro Nacional", f"${prediccion:.6f}")
                        
                        # Calcular precio mensual por vaca (estimado)
                        litros_vaca_mes = 500  # litros promedio por vaca por mes
                        ingreso_vaca_mes = prediccion * litros_vaca_mes
                        st.metric("Ingreso estimado por vaca/mes", f"${ingreso_vaca_mes:,.0f}")
                        
                        st.write(f"**Tipo de Modelo:** {model_G.get('tipo_modelo', 'N/A')}")
                        st.write(f"**Preprocesamiento:** {model_G.get('preprocesamiento', 'N/A')}")
                    
                    with col_det2:
                        if 'metricas' in model_G:
                            st.write(f"**R² Esperado:** {model_G['metricas'].get('r2', 'N/A'):.6f}")
                            st.write(f"**MSE Esperado:** {model_G['metricas'].get('mse', 'N/A'):.6f}")
                        
                        # Análisis de variables macroeconómicas
                        if dolar_oficial > 0:
                            precio_usd = prediccion / dolar_oficial
                            st.metric("Precio en USD", f"USD {precio_usd:.4f}/litro")
                    
                    # Análisis del contexto macroeconómico
                    st.subheader("🔍 Análisis Macroeconómico")
                    col_analisis1, col_analisis2 = st.columns(2)
                    
                    with col_analisis1:
                        if ipc_mensual > 8000:
                            st.error("🔴 IPC Alto: Alta inflación acumulada")
                        elif ipc_mensual > 5000:
                            st.warning("🟡 IPC Moderado: Inflación controlada")
                        else:
                            st.success("🟢 IPC Bajo: Estabilidad de precios")
                    
                    with col_analisis2:
                        if dolar_oficial > 1500:
                            st.warning("💰 Dólar Alto: Impacto en costos de insumos")
                        elif dolar_oficial > 1000:
                            st.info("📊 Dólar Moderado: Nivel intermedio")
                        else:
                            st.success("💚 Dólar Estable: Favorece estabilidad de costos")
                    
                    # Tendencias y proyecciones
                    st.subheader("📈 Interpretación Económica")
                    st.write("**Correlación IPC-Precio:** El IPC es un fuerte predictor del precio de la leche")
                    st.write("**Impacto Cambiario:** El dólar oficial afecta costos de insumos y competitividad")
                    st.write("**Modelo Simplificado:** Este modelo usa solo 2 variables pero logra alta precisión")
                    
                else:
                    # Es solo el modelo sklearn (formato antiguo)
                    st.warning("⚠️ Modelo en formato legacy. Aplicando predicción directa.")
                    datos_entrada = np.array([valores])
                    prediccion = model_G.predict(datos_entrada)
                    st.success(f"🥛 **Precio de Leche Predicho: ${prediccion[0]:.6f}/litro**")
                
                # Interpretación de resultados
                st.subheader("📊 Interpretación del Precio")
                if prediccion > 150:
                    st.error("🔴 Precio Alto: Por encima del promedio histórico")
                elif prediccion > 100:
                    st.warning("🟡 Precio Moderado: Dentro del rango normal")
                elif prediccion > 50:
                    st.success("🟢 Precio Competitivo: Favorable para el sector")
                else:
                    st.info("🔵 Precio Bajo: Verificar condiciones del mercado")
                
                # Escenarios económicos
                st.subheader("🎯 Escenarios Económicos")
                st.write("💡 **Escenario Inflacionario:** IPC alto genera aumentos de precios lácteos")
                st.write("💱 **Escenario Cambiario:** Dólar alto encarece insumos pero mejora exportaciones")
                st.write("⚖️ **Equilibrio:** Balance entre inflación y tipo de cambio determina rentabilidad")
                
            except Exception as e:
                st.error(f"❌ Error en la predicción: {str(e)}")
                
                # Información de debug
                with st.expander("🔧 Información de Debug"):
                    st.write("**Valores ingresados:**", valores)
                    st.write("**Tipo de modelo:**", type(model_G))
                    st.write("**Detalles del error:**", str(e))
    
    # Información contextual
    st.subheader("📋 Información Macroeconómica")
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.info("""
        **IPC (Índice de Precios al Consumidor):**
        - Mide la evolución de precios de bienes y servicios
        - Indicador principal de inflación
        - Base 2016 = 100 (serie actual)
        - Publicado mensualmente por INDEC
        """)
    
    with col_info2:
        st.info("""
        **Dólar Oficial:**
        - Cotización oficial del BCRA
        - Usado para operaciones comerciales
        - Impacta en costos de importación
        - Afecta competitividad exportadora
        """)
    
    # Ventajas del modelo simplificado
    st.subheader("✨ Ventajas del Modelo Macroeconómico")
    st.success("""
    **🎯 Simplicidad:** Solo 2 variables fáciles de obtener
    **📊 Alta Precisión:** R² > 0.999 con variables básicas
    **🔄 Actualización Rápida:** Datos disponibles mensualmente
    **📈 Interpretación Clara:** Relación directa inflación-precios
    **🌐 Visión Macro:** Se enfoca en tendencias económicas generales
    """)
    
    # Mostrar información adicional
    st.subheader("💡 Consejos de Uso")
    st.info("""
    **Para obtener mejores predicciones:**
    - Use el IPC más reciente disponible del INDEC
    - El IPC debe ser acumulado, no el porcentaje mensual
    - Use el tipo de cambio oficial vigente
    - Considere tendencias trimestrales para mejor contexto
    - Este modelo es ideal para análisis macroeconómicos rápidos
    """)