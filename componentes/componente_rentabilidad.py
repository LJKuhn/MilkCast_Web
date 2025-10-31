def C_rentabilidad(model_A):
    import streamlit as st
    import numpy as np
    import pandas as pd
    
    st.header("🎯 Predicción de Rentabilidad")
    
    # Descripción del modelo
    st.write("Predice la rentabilidad basándose en costos, precios, variables macroeconómicas y promedios sectoriales.")
    
    # Información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        st.write("**Tipo de Modelo:** RandomForest")
        st.write("**Preprocesamiento:** Estandarización (StandardScaler)")
        st.write("**R² Esperado:** 0.831300")
        st.write("**Variables utilizadas:**")
        st.write("- COSTO")
        st.write("- Precio/litro Nacional - SIGLeA") 
        st.write("- DOLAR OFICIAL $/US$")
        st.write("- IPC-Mensual")
        st.write("- IPIM Nivel General - INDEC")
        st.write("- Promedio del sector")
    
    # Crear columnas para inputs
    col1, col2 = st.columns(2)
    
    with col1:
        costo = st.number_input(
            label="💰 COSTO (ARS)",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_costo_rentabilidad",
            help="Costo de producción por litro"
        )
        
        precio_litro = st.number_input(
            label="🥛 Precio/litro Nacional - SIGLeA (ARS)",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_precio_litro_rentabilidad",
            help="Precio nacional por litro según SIGLeA"
        )
        
        dolar_oficial = st.number_input(
            label="💵 DOLAR OFICIAL $/US$ (ARS)",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_dolar_rentabilidad",
            help="Tipo de cambio oficial USD/ARS"
        )
    
    with col2:
        ipc_mensual = st.number_input(
            label="📊 IPC-Mensual",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_ipc_rentabilidad",
            help="Índice de Precios al Consumidor mensual"
        )
        
        ipim = st.number_input(
            label="🏭 IPIM Nivel General - INDEC",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_ipim_rentabilidad",
            help="Índice de Precios Internos al por Mayor"
        )
        
        promedio_sector = st.number_input(
            label="📈 Promedio del sector",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_promedio_rentabilidad",
            help="Promedio general del sector lácteo"
        )
    
    # Función para hacer predicción con manejo correcto del scaler
    def hacer_prediccion_rentabilidad(modelo_completo, valores):
        # Crear DataFrame con nombres de columnas correctos
        feature_names = ['COSTO', 'Precio/litro Nacional - SIGLeA', 'DOLAR OFICIAL $/US$', 
                        'IPC-Mensual', 'IPIM Nivel General - INDEC', 'Promedio del sector']
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
    if st.button("🔮 Predecir Rentabilidad", key="boton_rentabilidad"):
        # Validar que no haya valores negativos
        valores = [costo, precio_litro, dolar_oficial, ipc_mensual, ipim, promedio_sector]
        
        if any(v < 0 for v in valores):
            st.error("❌ Por favor, ingrese solo valores positivos.")
        elif any(v == 0 for v in valores):
            st.warning("⚠️ Algunos valores son cero. Verifique que sean correctos.")
        else:
            try:
                # Verificar si el modelo es el formato completo PKL o solo el modelo sklearn
                if isinstance(model_A, dict) and 'modelo' in model_A:
                    # Es el formato completo PKL
                    prediccion = hacer_prediccion_rentabilidad(model_A, valores)
                    st.success(f"🎯 **Rentabilidad Predicha: {prediccion:.6f}**")
                    
                    # Mostrar detalles adicionales
                    st.subheader("📊 Detalles de la Predicción")
                    col_det1, col_det2 = st.columns(2)
                    
                    with col_det1:
                        st.metric("Rentabilidad", f"{prediccion:.6f}")
                        st.write(f"**Tipo de Modelo:** {model_A.get('tipo_modelo', 'N/A')}")
                        st.write(f"**Preprocesamiento:** {model_A.get('preprocesamiento', 'N/A')}")
                    
                    with col_det2:
                        if 'metricas' in model_A:
                            st.write(f"**R² Esperado:** {model_A['metricas'].get('r2', 'N/A'):.6f}")
                            st.write(f"**MSE Esperado:** {model_A['metricas'].get('mse', 'N/A'):.6f}")
                    
                else:
                    # Es solo el modelo sklearn (formato antiguo)
                    st.warning("⚠️ Modelo en formato legacy. Aplicando predicción directa.")
                    datos_entrada = np.array([valores])
                    prediccion = model_A.predict(datos_entrada)
                    st.success(f"🎯 **Rentabilidad Predicha: {prediccion[0]:.6f}**")
                
                # Interpretación de resultados
                st.subheader("📈 Interpretación")
                if prediccion > 0.5:
                    st.success("✅ Rentabilidad Alta: Condiciones favorables para la producción")
                elif prediccion > 0.2:
                    st.warning("⚠️ Rentabilidad Media: Monitorear condiciones del mercado")
                else:
                    st.error("❌ Rentabilidad Baja: Evaluar estrategias de optimización")
                
            except Exception as e:
                st.error(f"❌ Error en la predicción: {str(e)}")
                
                # Información de debug
                with st.expander("🔧 Información de Debug"):
                    st.write("**Valores ingresados:**", valores)
                    st.write("**Tipo de modelo:**", type(model_A))
                    st.write("**Detalles del error:**", str(e))
    
    # Mostrar información adicional
    st.subheader("💡 Consejos de Uso")
    st.info("""
    **Para obtener mejores predicciones:**
    - Utilice datos actualizados y precisos
    - El IPC debe ser el valor acumulado del período
    - El dólar oficial debe ser el tipo de cambio vigente
    - Los costos deben incluir todos los factores de producción
    - Verifique que los valores estén en las unidades correctas
    """)