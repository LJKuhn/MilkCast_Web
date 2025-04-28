def C_prediccion(model1):
    import streamlit as st
    import numpy as np
    import pickle

    st.header("Prediccion en base a IPC y valor del dolar")

    # Descripción
    st.write("Introducir el valor del IPC y del dolar para predecir el valor del litro de leche, intente introducir un valor que cuente con decimales.")

    # Crear dos columnas para el input
    col1, col2 = st.columns(2)

    with col1:
        ipc = st.number_input(
            label="Ingrese el valor del IPC",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            key="input_ipc"
        )

    with col2:
        dolar = st.number_input(
            label="Ingrese el valor del Dólar (ARS)",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            key="input_dolar"
        )

    # Botón para realizar la predicción
    if st.button("Realizar Predicción", key="boton_prediccion"):
        # Asegurarse de que los valores no sean negativos
        if ipc < 0 or dolar < 0:
            st.error("Por favor, ingrese valores válidos (positivos) para IPC y Dólar.")
        else:
            # Preparar los datos para el modelo
            datos_entrada = np.array([[ipc, dolar]])  # Los valores deben estar en un array bidimensional

            # Hacer la predicción
            try:
                prediccion = model1.predict(datos_entrada)  # Hacer la predicción usando el modelo
                st.success(f"🔮 El precio predicho de la leche es: **${prediccion[0]:,.2f}**")
            except Exception as e:
                st.error(f"Error en la predicción: {e}")

