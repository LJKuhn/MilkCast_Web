def C_costos(model_B):
    import streamlit as st
    import numpy as np
    import pandas as pd
    
    st.header("💰 Predicción de Costos")
    
    # Descripción del modelo
    st.write("Predice los costos de producción láctea basándose en promedios sectoriales, relaciones económicas y variables macroeconómicas.")
    
    # Información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        st.write("**Tipo de Modelo:** RandomForest")
        st.write("**Preprocesamiento:** Datos Originales (sin estandarización)")
        st.write("**R² Esperado:** 0.999726")
        st.write("**Variables utilizadas:**")
        st.write("- Promedio del sector")
        st.write("- RELACION LECHE/MAIZ") 
        st.write("- IPIM Nivel General - INDEC")
        st.write("- DOLAR OFICIAL $/US$")
        st.write("- RELACION VAQUILLONA AL PARIR - LECHE")
        st.write("- IPC - INDEC CoberNac")
    
    # Crear columnas para inputs
    col1, col2 = st.columns(2)
    
    with col1:
        promedio_sector = st.number_input(
            label="📈 Promedio del sector",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_promedio_sector_costos",
            help="Promedio general del sector lácteo"
        )
        
        relacion_leche_maiz = st.number_input(
            label="🌽 RELACION LECHE/MAIZ",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_relacion_leche_maiz",
            help="Relación de precios entre leche y maíz"
        )
        
        ipim_general = st.number_input(
            label="🏭 IPIM Nivel General - INDEC",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_ipim_costos",
            help="Índice de Precios Internos al por Mayor"
        )
    
    with col2:
        dolar_oficial = st.number_input(
            label="💵 DOLAR OFICIAL $/US$ (ARS)",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_dolar_costos",
            help="Tipo de cambio oficial USD/ARS"
        )
        
        relacion_vaquillona_leche = st.number_input(
            label="🐄 RELACION VAQUILLONA AL PARIR - LECHE",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_relacion_vaquillona",
            help="Relación entre precio de vaquillona y leche"
        )
        
        ipc_cobernac = st.number_input(
            label="📊 IPC - INDEC CoberNac",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_ipc_cobernac",
            help="IPC con cobertura nacional"
        )
    
    # Función para hacer predicción con manejo correcto del scaler
    def hacer_prediccion_costos(modelo_completo, valores):
        # Crear DataFrame con nombres de columnas correctos
        feature_names = ['Promedio del sector', 'RELACION LECHE/MAIZ', 'IPIM Nivel General - INDEC',
                        'DOLAR OFICIAL $/US$', 'RELACION VAQUILLONA AL PARIR - LECHE', 'IPC - INDEC CoberNac']
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
    if st.button("🔮 Predecir Costos", key="boton_costos"):
        # Validar que no haya valores negativos
        valores = [promedio_sector, relacion_leche_maiz, ipim_general, dolar_oficial, relacion_vaquillona_leche, ipc_cobernac]
        
        if any(v < 0 for v in valores):
            st.error("❌ Por favor, ingrese solo valores positivos.")
        elif any(v == 0 for v in valores):
            st.warning("⚠️ Algunos valores son cero. Verifique que sean correctos.")
        else:
            try:
                # Verificar si el modelo es el formato completo PKL o solo el modelo sklearn
                if isinstance(model_B, dict) and 'modelo' in model_B:
                    # Es el formato completo PKL
                    prediccion = hacer_prediccion_costos(model_B, valores)
                    st.success(f"💰 **Costo Predicho: ${prediccion:.6f}**")
                    
                    # Mostrar detalles adicionales
                    st.subheader("📊 Detalles de la Predicción")
                    col_det1, col_det2 = st.columns(2)
                    
                    with col_det1:
                        st.metric("Costo de Producción", f"${prediccion:.6f}")
                        st.write(f"**Tipo de Modelo:** {model_B.get('tipo_modelo', 'N/A')}")
                        st.write(f"**Preprocesamiento:** {model_B.get('preprocesamiento', 'N/A')}")
                    
                    with col_det2:
                        if 'metricas' in model_B:
                            st.write(f"**R² Esperado:** {model_B['metricas'].get('r2', 'N/A'):.6f}")
                            st.write(f"**MSE Esperado:** {model_B['metricas'].get('mse', 'N/A'):.6f}")
                    
                else:
                    # Es solo el modelo sklearn (formato antiguo)
                    st.warning("⚠️ Modelo en formato legacy. Aplicando predicción directa.")
                    datos_entrada = np.array([valores])
                    prediccion = model_B.predict(datos_entrada)
                    st.success(f"💰 **Costo Predicho: ${prediccion[0]:.6f}**")
                
                # Interpretación de resultados
                st.subheader("📈 Interpretación")
                if prediccion > 100:
                    st.error("🔴 Costos Altos: Evaluar estrategias de reducción de costos")
                elif prediccion > 50:
                    st.warning("🟡 Costos Moderados: Monitorear tendencias del mercado")
                else:
                    st.success("🟢 Costos Controlados: Condiciones favorables de producción")
                
                # Análisis de factores influyentes
                st.subheader("🔍 Factores Más Influyentes")
                st.write("**Relación Leche/Maíz:** Impacta directamente en costos de alimentación")
                st.write("**Dólar Oficial:** Afecta costos de insumos importados")
                st.write("**IPIM:** Refleja inflación en precios al por mayor")
                
            except Exception as e:
                st.error(f"❌ Error en la predicción: {str(e)}")
                
                # Información de debug
                with st.expander("🔧 Información de Debug"):
                    st.write("**Valores ingresados:**", valores)
                    st.write("**Tipo de modelo:**", type(model_B))
                    st.write("**Detalles del error:**", str(e))
    
    # Mostrar información adicional
    st.subheader("💡 Consejos de Uso")
    st.info("""
    **Para obtener mejores predicciones de costos:**
    - Mantenga actualizados los valores del IPIM e IPC
    - La relación leche/maíz es clave para costos de alimentación
    - Considere el impacto del dólar en insumos importados
    - El promedio sectorial debe reflejar condiciones actuales
    - Las relaciones de precios deben ser consistentes temporalmente
    """)