def C_precio_queso(model_D):
    import streamlit as st
    import numpy as np
    import pandas as pd
    
    st.header("🧀 Predicción de Precio de Queso")
    
    # Descripción del modelo
    st.write("Predice el precio del queso tipo cuartirolo basándose en precios lácteos, índices económicos y elaboración total.")
    
    # Información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        st.write("**Tipo de Modelo:** LinearRegression")
        st.write("**Preprocesamiento:** Estandarización (StandardScaler)")
        st.write("**R² Esperado:** 0.999357")
        st.write("**Variables utilizadas:**")
        st.write("- Precio/litro Nacional - SIGLeA")
        st.write("- IPC-Mensual") 
        st.write("- IPIM Lácteos - INDEC")
        st.write("- ELABORACIÓN TOTAL")
        st.write("- Promedio general sector privado")
    
    # Crear columnas para inputs
    col1, col2 = st.columns(2)
    
    with col1:
        precio_litro = st.number_input(
            label="🥛 Precio/litro Nacional - SIGLeA (ARS)",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_precio_litro_queso",
            help="Precio nacional por litro según SIGLeA"
        )
        
        ipc_mensual = st.number_input(
            label="📊 IPC-Mensual",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_ipc_queso",
            help="Índice de Precios al Consumidor mensual"
        )
        
        ipim_lacteos = st.number_input(
            label="🏭 IPIM Lácteos - INDEC",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_ipim_lacteos",
            help="Índice de Precios Internos al por Mayor - Sector Lácteos"
        )
    
    with col2:
        elaboracion_total = st.number_input(
            label="🏭 ELABORACIÓN TOTAL",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_elaboracion_total",
            help="Volumen total de elaboración láctea"
        )
        
        promedio_sector_privado = st.number_input(
            label="📈 Promedio general sector privado",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_promedio_privado",
            help="Promedio de precios del sector privado"
        )
    
    # Función para hacer predicción con manejo correcto del scaler
    def hacer_prediccion_precio_queso(modelo_completo, valores):
        # Crear DataFrame con nombres de columnas correctos
        feature_names = ['Precio/litro Nacional - SIGLeA', 'IPC-Mensual', 'IPIM Lácteos - INDEC',
                        'ELABORACIÓN TOTAL', 'Promedio general sector privado']
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
    if st.button("🔮 Predecir Precio de Queso", key="boton_precio_queso"):
        # Validar que no haya valores negativos
        valores = [precio_litro, ipc_mensual, ipim_lacteos, elaboracion_total, promedio_sector_privado]
        
        if any(v < 0 for v in valores):
            st.error("❌ Por favor, ingrese solo valores positivos.")
        elif any(v == 0 for v in valores):
            st.warning("⚠️ Algunos valores son cero. Verifique que sean correctos.")
        else:
            try:
                # Verificar si el modelo es el formato completo PKL o solo el modelo sklearn
                if isinstance(model_D, dict) and 'modelo' in model_D:
                    # Es el formato completo PKL
                    prediccion = hacer_prediccion_precio_queso(model_D, valores)
                    st.success(f"🧀 **Precio de Queso Predicho: ${prediccion:.2f}/kg**")
                    
                    # Mostrar detalles adicionales
                    st.subheader("📊 Detalles de la Predicción")
                    col_det1, col_det2 = st.columns(2)
                    
                    with col_det1:
                        st.metric("Precio Queso Cuartirolo", f"${prediccion:.2f}/kg")
                        st.write(f"**Tipo de Modelo:** {model_D.get('tipo_modelo', 'N/A')}")
                        st.write(f"**Preprocesamiento:** {model_D.get('preprocesamiento', 'N/A')}")
                    
                    with col_det2:
                        if 'metricas' in model_D:
                            st.write(f"**R² Esperado:** {model_D['metricas'].get('r2', 'N/A'):.6f}")
                            st.write(f"**MSE Esperado:** {model_D['metricas'].get('mse', 'N/A'):.6f}")
                    
                    # Comparación con precio de leche
                    if precio_litro > 0:
                        relacion_queso_leche = prediccion / precio_litro
                        st.subheader("🔍 Análisis de Relación")
                        st.write(f"**Relación Precio Queso/Leche:** {relacion_queso_leche:.2f}")
                        
                        if relacion_queso_leche > 15:
                            st.info("📈 Relación alta: El queso tiene un premium considerable sobre la leche")
                        elif relacion_queso_leche > 10:
                            st.info("📊 Relación normal: Precio dentro de rangos esperados")
                        else:
                            st.warning("📉 Relación baja: Verificar consistencia de precios")
                    
                else:
                    # Es solo el modelo sklearn (formato antiguo)
                    st.warning("⚠️ Modelo en formato legacy. Aplicando predicción directa.")
                    datos_entrada = np.array([valores])
                    prediccion = model_D.predict(datos_entrada)
                    st.success(f"🧀 **Precio de Queso Predicho: ${prediccion[0]:.2f}/kg**")
                
                # Interpretación de resultados
                st.subheader("📈 Interpretación del Precio")
                if prediccion > 3000:
                    st.error("🔴 Precio Alto: Por encima del mercado promedio")
                elif prediccion > 2000:
                    st.warning("🟡 Precio Moderado: Dentro del rango normal")
                elif prediccion > 1000:
                    st.success("🟢 Precio Competitivo: Buena oportunidad de mercado")
                else:
                    st.info("🔵 Precio Bajo: Verificar condiciones del mercado")
                
            except Exception as e:
                st.error(f"❌ Error en la predicción: {str(e)}")
                
                # Información de debug
                with st.expander("🔧 Información de Debug"):
                    st.write("**Valores ingresados:**", valores)
                    st.write("**Tipo de modelo:**", type(model_D))
                    st.write("**Detalles del error:**", str(e))
    
    # Información contextual
    st.subheader("📋 Información del Producto")
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.info("""
        **Queso Tipo Cuartirolo:**
        - Queso fresco de pasta blanda
        - Maduración de 15-30 días
        - Contenido graso mínimo 45%
        - Producto lácteo de consumo masivo
        """)
    
    with col_info2:
        st.info("""
        **Factores de Precio:**
        - Precio de la leche cruda
        - Costos de elaboración
        - Índices de inflación
        - Demanda del mercado
        """)
    
    # Mostrar información adicional
    st.subheader("💡 Consejos de Uso")
    st.info("""
    **Para obtener mejores predicciones:**
    - Use precios actualizados de leche como base
    - El IPC mensual debe ser el valor acumulado
    - IPIM Lácteos refleja costos específicos del sector
    - La elaboración total indica la capacidad instalada
    - Considere estacionalidad en la demanda de quesos
    """)