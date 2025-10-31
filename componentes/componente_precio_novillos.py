def C_precio_novillos(model_F):
    import streamlit as st
    import numpy as np
    import pandas as pd
    
    st.header("🐄 Predicción de Precio de Novillos")
    
    # Descripción del modelo
    st.write("Predice el precio promedio de novillos basándose en la cantidad de vaquillonas, variables macroeconómicas y precios relacionados.")
    
    # Información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        st.write("**Tipo de Modelo:** Regresión Lineal")
        st.write("**Preprocesamiento:** Estandarización (StandardScaler)")
        st.write("**Variables utilizadas:**")
        st.write("- Cabezas Vaquillonas")
        st.write("- DOLAR OFICIAL $/US$") 
        st.write("- IPC-Mensual")
        st.write("- Precio Promedio Vaquillonas")
    
    # Crear columnas para inputs
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    with col1:
        cabezas_vaquillonas = st.number_input(
            label="🐄 Cabezas Vaquillonas",
            min_value=0.0,
            step=1.0,
            format="%.0f",
            key="input_cabezas_vaquillonas",
            help="Cantidad de cabezas de vaquillonas en el mercado (oferta)"
        )
    
    with col2:
        dolar_oficial = st.number_input(
            label="� DOLAR OFICIAL $/US$ (ARS)",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_dolar_novillos",
            help="Tipo de cambio oficial USD/ARS"
        )
    
    with col3:
        ipc_mensual = st.number_input(
            label="📊 IPC-Mensual",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_ipc_novillos",
            help="Índice de Precios al Consumidor mensual"
        )
    
    with col4:
        precio_vaquillonas = st.number_input(
            label="� Precio Promedio Vaquillonas (ARS)",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_precio_vaquillonas",
            help="Precio promedio de vaquillonas como referencia del mercado"
        )
    
    # Función para hacer predicción con manejo correcto del scaler
    def hacer_prediccion_precio_novillos(modelo_completo, valores):
        # Crear DataFrame con nombres de columnas correctos
        feature_names = ['Cabezas Vaquillonas', 'DOLAR OFICIAL $/US$', 'IPC-Mensual', 'Precio Promedio Vaquillonas']
        df_entrada = pd.DataFrame([valores], columns=feature_names)
        
        # Aplicar preprocesamiento si es necesario
        if modelo_completo.get('scaler') is not None:
            st.info("✅ Aplicando estandarización (StandardScaler)")
            datos_procesados = modelo_completo['scaler'].transform(df_entrada)
            # Convertir de vuelta a DataFrame para mantener nombres de columnas
            datos_procesados = pd.DataFrame(datos_procesados, columns=feature_names)
        else:
            st.info("✅ Usando datos originales (sin preprocesamiento)")
            datos_procesados = df_entrada
        
        # Hacer predicción
        prediccion = modelo_completo['modelo'].predict(datos_procesados)
        return prediccion[0]
    
    # Botón para realizar predicción
    if st.button("🔮 Predecir Precio de Novillos", key="boton_precio_novillos"):
        # Validar que no haya valores negativos
        valores = [cabezas_vaquillonas, dolar_oficial, ipc_mensual, precio_vaquillonas]
        
        if any(v < 0 for v in valores):
            st.error("❌ Por favor, ingrese solo valores positivos.")
        elif any(v == 0 for v in valores):
            st.warning("⚠️ Algunos valores son cero. Verifique que sean correctos.")
        else:
            try:
                # Verificar si el modelo es el formato completo PKL o solo el modelo sklearn
                if isinstance(model_F, dict) and 'modelo' in model_F:
                    # Es el formato completo PKL
                    prediccion = hacer_prediccion_precio_novillos(model_F, valores)
                    st.success(f"🐄 **Precio de Novillos Predicho: ${prediccion:.2f}**")
                    
                    # Mostrar detalles adicionales
                    st.subheader("📊 Detalles de la Predicción")
                    col_det1, col_det2 = st.columns(2)
                    
                    with col_det1:
                        st.metric("Precio Promedio Novillos", f"${prediccion:.2f}")
                        # Calcular precio por kg estimado (considerando peso promedio)
                        peso_estimado_novillo = 420  # kg promedio
                        precio_por_kg = prediccion / peso_estimado_novillo
                        st.metric("Precio Estimado por Kg", f"${precio_por_kg:.2f}/kg")
                        st.write(f"**Tipo de Modelo:** {model_F.get('tipo_modelo', 'N/A')}")
                        st.write(f"**Preprocesamiento:** {model_F.get('preprocesamiento', 'N/A')}")
                    
                    with col_det2:
                        if 'metricas' in model_F:
                            st.write(f"**R² Esperado:** {model_F['metricas'].get('r2', 'N/A'):.6f}")
                            st.write(f"**MSE Esperado:** {model_F['metricas'].get('mse', 'N/A'):.6f}")
                        
                        # Comparación con vaquillonas
                        if precio_vaquillonas > 0:
                            diferencia_precio = prediccion - precio_vaquillonas
                            porcentaje_diferencia = (diferencia_precio / precio_vaquillonas) * 100
                            st.write(f"**Diferencia con Vaquillonas:** ${diferencia_precio:.2f}")
                            st.write(f"**Porcentaje:** {porcentaje_diferencia:.1f}%")
                    
                    # Análisis del mercado ganadero
                    st.subheader("🔍 Análisis del Mercado Ganadero")
                    col_analisis1, col_analisis2 = st.columns(2)
                    
                    with col_analisis1:
                        if cabezas_vaquillonas > 80000:
                            st.info("📈 Oferta Alta: Gran cantidad de vaquillonas disponibles")
                        elif cabezas_vaquillonas > 40000:
                            st.info("📊 Oferta Normal: Mercado equilibrado")
                        else:
                            st.warning("📉 Oferta Baja: Escasa disponibilidad de vaquillonas")
                    
                    with col_analisis2:
                        if precio_vaquillonas > 0:
                            if diferencia_precio < -100000:
                                st.success("💚 Novillos más baratos: Buena oportunidad de compra")
                            elif diferencia_precio > 100000:
                                st.warning("🟡 Novillos más caros: Evaluar inversión")
                            else:
                                st.info("📊 Precios similares: Mercado estable")
                    
                else:
                    # Es solo el modelo sklearn (formato antiguo)
                    st.warning("⚠️ Modelo en formato legacy. Aplicando predicción directa.")
                    datos_entrada = np.array([valores])
                    prediccion = model_F.predict(datos_entrada)
                    st.success(f"🐄 **Precio de Novillos Predicho: ${prediccion[0]:.2f}**")
                
                # Interpretación de resultados
                st.subheader("📈 Interpretación del Precio")
                if prediccion > 800000:
                    st.error("🔴 Precio Alto: Por encima del mercado promedio")
                elif prediccion > 500000:
                    st.warning("🟡 Precio Moderado: Dentro del rango normal")
                elif prediccion > 300000:
                    st.success("🟢 Precio Competitivo: Buena oportunidad")
                else:
                    st.info("🔵 Precio Bajo: Verificar condiciones del mercado")
                
                # Recomendaciones
                st.subheader("💡 Recomendaciones")
                if cabezas_vaquillonas > 60000 and precio_vaquillonas > 0:
                    st.write("🎯 **Para Compradores:** Alta oferta puede significar mejores precios")
                    st.write("📊 **Para Vendedores:** Evaluar momento óptimo según demanda")
                elif cabezas_vaquillonas < 20000:
                    st.write("⚠️ **Oferta Limitada:** Los precios pueden incrementarse")
                    st.write("🏃 **Acción Rápida:** Considerar compras anticipadas")
                
            except Exception as e:
                st.error(f"❌ Error en la predicción: {str(e)}")
                
                # Información de debug
                with st.expander("🔧 Información de Debug"):
                    st.write("**Valores ingresados:**", valores)
                    st.write("**Tipo de modelo:**", type(model_F))
                    st.write("**Detalles del error:**", str(e))
    
    # Información contextual
    st.subheader("📋 Información del Mercado Ganadero")
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.info("""
        **Novillos en el Mercado:**
        - Animales de 18-24 meses
        - Peso promedio: 380-450 kg
        - Categoría de terminación
        - Precio influenciado por peso y calidad
        """)
    
    with col_info2:
        st.info("""
        **Factores de Precio:**
        - Oferta de vaquillonas (reposición)
        - Variables macroeconómicas (IPC, USD)
        - Precios de categorías relacionadas
        - Demanda de exportación
        """)
    
    # Mostrar información adicional
    st.subheader("💡 Consejos de Uso")
    st.info("""
    **Para obtener mejores predicciones:**
    - Use datos actualizados de remates y ferias
    - La cantidad de vaquillonas refleja la oferta futura
    - El IPC mensual debe ser el más reciente
    - El dólar oficial impacta en la competitividad exportadora
    - Los precios de vaquillonas son un indicador clave del mercado
    """)