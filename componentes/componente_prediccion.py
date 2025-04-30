def C_prediccion(model1):
    import streamlit as st
    import numpy as np
    import pickle

    st.header("Prediccion en base a IPC y valor del dolar")

    # Descripción
    st.write("Introducir el valor del IPC y del dolar para predecir el valor del litro de leche, intente introducir un valor que cuente con decimales.")

    st.markdown("### Instrucciones para ingresar el IPC como variable predictora")

    st.info("""
    Para usar el IPC como variable predictora en el modelo, ingresá el valor acumulado correspondiente al mes de cada fila.

    - La serie histórica comienza en un valor base de **98**.
    - El último valor oficial disponible es **7864,1257** para **enero de 2025**.
    - Para obtener el IPC de **febrero de 2025**, multiplicá: `7864,1257 × 1,021` (2,1% de inflación).
    - Para **marzo de 2025**, usá: `7864,1257 × 1,021 × 1,037` (2,1% en febrero + 3,7% en marzo, se puede utilizar tambien para principios de abril).
    - Para **abril de 2025**, deberás continuar multiplicando por el coeficiente del mes correspondiente (cuando esté disponible).

    ⚠️ Recordá: el valor que ingresás debe ser **acumulado**, no solo el porcentaje mensual.
    """)

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

