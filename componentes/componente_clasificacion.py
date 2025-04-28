def C_clasificacion(model2):
    import streamlit as st
    import numpy as np
    import pickle

    st.header("Prediccion en base a precio Leche Entera, Queso y Yogur")

    # Descripción
    st.write("Introducir el valor de Leche Entera, Queso y Yogur para predecir el valor del litro de leche, intente introducir un valor que cuente con decimales.")

    # Crear dos columnas para el input
    col1, col2, col3 = st.columns(3)

    with col1:
        LECHE_COMUN = st.number_input(
            label=" LECHE COMUN ENTERA $/litro (ARS)",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            key="input_ LECHE_COMUN"
        )

    with col2:
        QUESO = st.number_input(
            label="QUESO TIPO CUARTIROLO $/kg (ARS)",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            key="input_QUESO"
        )
    
    with col3:
        YOGUR = st.number_input(
            label="YOGUR para beber sachet $/1000 grs (ARS)",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            key="input_YOGUR"
        )

    # Botón para realizar la predicción
    if st.button("Realizar Predicción", key="boton_prediccion1"):
        # Asegurarse de que los valores no sean negativos
        if  LECHE_COMUN < 0 or QUESO < 0 or YOGUR < 0:
            st.error("Por favor, ingrese valores válidos (positivos) para IPC y Dólar.")
        else:
            # Preparar los datos para el modelo
            datos_entrada = np.array([[LECHE_COMUN, QUESO, YOGUR]])  # Los valores deben estar en un array bidimensional

            # Hacer la predicción
            try:
                prediccion = model2.predict(datos_entrada)  # Hacer la predicción usando el modelo
                st.success(f"🔮 El precio predicho de la leche es: **${prediccion[0]:,.2f}**")
            except Exception as e:
                st.error(f"Error en la predicción: {e}")







