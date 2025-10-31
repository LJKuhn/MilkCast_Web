def C_productos_lacteos(model_H):
    import streamlit as st
    import numpy as np
    import pandas as pd
    
    st.header("🥛 Predicción con Productos Lácteos Específicos")
    
    # Descripción del modelo
    st.write("Predice el precio de la leche basándose en precios minoristas de productos lácteos específicos del mercado.")
    
    # Información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        st.write("**Tipo de Modelo:** RandomForest")
        st.write("**Preprocesamiento:** Datos Originales (sin estandarización)")
        st.write("**R² Esperado:** 0.999820")
        st.write("**Variables utilizadas:**")
        st.write("- LECHE COMUN ENTERA $/litro")
        st.write("- QUESO TIPO CUARTIROLO $/kg")
        st.write("- YOGUR para beber sachet $/1000 grs")
    
    # Crear columnas para inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        leche_entera = st.number_input(
            label="🥛 LECHE COMÚN ENTERA $/litro",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_leche_entera_h",
            help="Precio minorista de leche común entera por litro"
        )
        
        # Información adicional
        with st.expander("ℹ️ Sobre Leche Entera"):
            st.write("""
            **Leche Común Entera:**
            - Contenido graso mínimo 3%
            - Producto de consumo masivo
            - Precio minorista en góndola
            - Indicador directo del mercado
            """)
    
    with col2:
        queso_cuartirolo = st.number_input(
            label="🧀 QUESO TIPO CUARTIROLO $/kg",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_queso_cuartirolo_h",
            help="Precio minorista de queso cuartirolo por kilogramo"
        )
        
        # Información adicional
        with st.expander("ℹ️ Sobre Queso Cuartirolo"):
            st.write("""
            **Queso Tipo Cuartirolo:**
            - Queso fresco de pasta blanda
            - Maduración 15-30 días
            - Gran consumo en Argentina
            - Indicador de productos elaborados
            """)
    
    with col3:
        yogur_sachet = st.number_input(
            label="🥤 YOGUR para beber sachet $/1000 grs",
            min_value=0.0,
            step=0.01,
            format="%.6f",
            key="input_yogur_sachet_h",
            help="Precio minorista de yogur para beber por 1000 gramos"
        )
        
        # Información adicional
        with st.expander("ℹ️ Sobre Yogur para Beber"):
            st.write("""
            **Yogur para Beber:**
            - Producto fermentado líquido
            - Formato sachet de 1000g
            - Mayor valor agregado
            - Indicador de productos premium
            """)
    
    # Función para hacer predicción con manejo correcto del scaler
    def hacer_prediccion_productos_lacteos(modelo_completo, valores):
        # Crear DataFrame con nombres de columnas correctos
        feature_names = ['LECHE COMUN ENTERA $/litro', 'QUESO TIPO CUARTIROLO $/kg', 'YOGUR para beber sachet $/1000 grs']
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
    if st.button("🔮 Predecir Precio con Productos Lácteos", key="boton_productos_lacteos"):
        # Validar que no haya valores negativos
        valores = [leche_entera, queso_cuartirolo, yogur_sachet]
        
        if any(v < 0 for v in valores):
            st.error("❌ Por favor, ingrese solo valores positivos.")
        elif any(v == 0 for v in valores):
            st.warning("⚠️ Algunos valores son cero. Verifique que sean correctos.")
        else:
            try:
                # Verificar si el modelo es el formato completo PKL o solo el modelo sklearn
                if isinstance(model_H, dict) and 'modelo' in model_H:
                    # Es el formato completo PKL
                    prediccion = hacer_prediccion_productos_lacteos(model_H, valores)
                    st.success(f"🥛 **Precio de Leche Predicho: ${prediccion:.6f}/litro**")
                    
                    # Mostrar detalles adicionales
                    st.subheader("📊 Detalles de la Predicción")
                    col_det1, col_det2 = st.columns(2)
                    
                    with col_det1:
                        st.metric("Precio/litro Nacional SIGLeA", f"${prediccion:.6f}")
                        
                        # Comparación con precio de leche entera ingresada
                        if leche_entera > 0:
                            diferencia = prediccion - leche_entera
                            porcentaje = (diferencia / leche_entera) * 100
                            st.metric("Diferencia con Leche Entera", f"${diferencia:.6f}")
                            st.write(f"**Porcentaje:** {porcentaje:.2f}%")
                        
                        st.write(f"**Tipo de Modelo:** {model_H.get('tipo_modelo', 'N/A')}")
                        st.write(f"**Preprocesamiento:** {model_H.get('preprocesamiento', 'N/A')}")
                    
                    with col_det2:
                        if 'metricas' in model_H:
                            st.write(f"**R² Esperado:** {model_H['metricas'].get('r2', 'N/A'):.6f}")
                            st.write(f"**MSE Esperado:** {model_H['metricas'].get('mse', 'N/A'):.6f}")
                    
                    # Análisis de la cadena láctea
                    st.subheader("🔍 Análisis de la Cadena Láctea")
                    col_analisis1, col_analisis2 = st.columns(2)
                    
                    with col_analisis1:
                        # Relaciones de precios
                        if leche_entera > 0 and queso_cuartirolo > 0:
                            relacion_queso_leche = queso_cuartirolo / leche_entera
                            st.write(f"**Relación Queso/Leche:** {relacion_queso_leche:.1f}")
                            if relacion_queso_leche > 20:
                                st.success("✅ Relación normal: Queso con buen margen")
                            else:
                                st.warning("⚠️ Relación baja: Queso poco diferenciado")
                    
                    with col_analisis2:
                        if yogur_sachet > 0 and leche_entera > 0:
                            relacion_yogur_leche = yogur_sachet / leche_entera
                            st.write(f"**Relación Yogur/Leche:** {relacion_yogur_leche:.1f}")
                            if relacion_yogur_leche > 3:
                                st.success("✅ Alto valor agregado en yogur")
                            else:
                                st.info("📊 Valor agregado moderado")
                    
                    # Análisis de consistencia de precios
                    st.subheader("📈 Análisis de Consistencia")
                    if all(v > 0 for v in valores):
                        # Verificar coherencia de precios
                        precio_esperado_queso = leche_entera * 15  # Relación típica
                        precio_esperado_yogur = leche_entera * 2.5  # Relación típica
                        
                        coherencia_queso = abs(queso_cuartirolo - precio_esperado_queso) / precio_esperado_queso
                        coherencia_yogur = abs(yogur_sachet - precio_esperado_yogur) / precio_esperado_yogur
                        
                        if coherencia_queso < 0.3:
                            st.success("✅ Precio de queso coherente con leche")
                        else:
                            st.warning("⚠️ Precio de queso puede estar desalineado")
                        
                        if coherencia_yogur < 0.3:
                            st.success("✅ Precio de yogur coherente con leche")
                        else:
                            st.warning("⚠️ Precio de yogur puede estar desalineado")
                    
                else:
                    # Es solo el modelo sklearn (formato antiguo)
                    st.warning("⚠️ Modelo en formato legacy. Aplicando predicción directa.")
                    datos_entrada = np.array([valores])
                    prediccion = model_H.predict(datos_entrada)
                    st.success(f"🥛 **Precio de Leche Predicho: ${prediccion[0]:.6f}/litro**")
                
                # Interpretación de resultados
                st.subheader("📊 Interpretación del Precio")
                if prediccion > 150:
                    st.error("🔴 Precio Alto: Mercado con alta valoración")
                elif prediccion > 100:
                    st.warning("🟡 Precio Moderado: Dentro del rango normal")
                elif prediccion > 50:
                    st.success("🟢 Precio Competitivo: Favorable para consumidores")
                else:
                    st.info("🔵 Precio Bajo: Verificar condiciones del mercado")
                
                # Insights del modelo
                st.subheader("💡 Insights del Modelo")
                st.write("🎯 **Ventaja:** Este modelo usa precios minoristas reales y actuales")
                st.write("📊 **Precisión:** Alta correlación con productos del mercado")
                st.write("🔄 **Actualización:** Precios fáciles de obtener en comercios")
                st.write("🏪 **Practicidad:** Ideal para análisis de mercado retail")
                
            except Exception as e:
                st.error(f"❌ Error en la predicción: {str(e)}")
                
                # Información de debug
                with st.expander("🔧 Información de Debug"):
                    st.write("**Valores ingresados:**", valores)
                    st.write("**Tipo de modelo:**", type(model_H))
                    st.write("**Detalles del error:**", str(e))
    
    # Calculadora de relaciones
    st.subheader("🧮 Calculadora de Relaciones de Precios")
    if all(v > 0 for v in [leche_entera, queso_cuartirolo, yogur_sachet]):
        col_calc1, col_calc2, col_calc3 = st.columns(3)
        
        with col_calc1:
            relacion_ql = queso_cuartirolo / leche_entera
            st.metric("Queso/Leche", f"{relacion_ql:.1f}x")
            if 12 <= relacion_ql <= 20:
                st.success("✅ Relación normal")
            else:
                st.warning("⚠️ Fuera de rango típico")
        
        with col_calc2:
            relacion_yl = yogur_sachet / leche_entera  
            st.metric("Yogur/Leche", f"{relacion_yl:.1f}x")
            if 2 <= relacion_yl <= 4:
                st.success("✅ Relación normal")
            else:
                st.warning("⚠️ Fuera de rango típico")
        
        with col_calc3:
            relacion_yq = yogur_sachet / queso_cuartirolo
            st.metric("Yogur/Queso", f"{relacion_yq:.2f}x")
            if 0.1 <= relacion_yq <= 0.3:
                st.success("✅ Relación normal")
            else:
                st.warning("⚠️ Fuera de rango típico")
    
    # Información contextual
    st.subheader("📋 Información de Productos")
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.info("""
        **Cadena de Valor Láctea:**
        - Leche cruda → Precio productor
        - Leche entera → Precio minorista
        - Queso → Valor agregado medio
        - Yogur → Alto valor agregado
        """)
    
    with col_info2:
        st.info("""
        **Factores de Precio:**
        - Costos de producción
        - Márgenes de comercialización
        - Demanda del consumidor
        - Competencia en góndola
        """)
    
    # Mostrar información adicional
    st.subheader("💡 Consejos de Uso")
    st.info("""
    **Para obtener mejores predicciones:**
    - Use precios actuales de supermercados o comercios
    - Verifique que los precios sean coherentes entre productos
    - Considere variaciones regionales en los precios
    - Este modelo es ideal para análisis de mercado minorista
    - Los precios deben corresponder al mismo período temporal
    """)