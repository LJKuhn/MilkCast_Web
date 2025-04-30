def C_visualizacion(df):
    import os
    import streamlit as st
    import pandas as pd
    import pickle
    import numpy as np
    import plotly.express as px
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import seaborn as sns
    from sklearn.preprocessing import MinMaxScaler
    from scipy import interpolate
    from scipy.ndimage import uniform_filter1d

    imagenes_dir = "imagenes"

    st.header("Visualización de Datos sobre la Producción Lechera")

    st.markdown(
        '''
        <div style="text-align: justify;">
            Con el auge del análisis de datos y el desarrollo de algoritmos orientados a facilitar la toma de decisiones, hemos decidido enfocar nuestros esfuerzos en brindar soluciones innovadoras a los productores agropecuarios. En un contexto de profundos cambios económicos —como la reactivación del crédito, el levantamiento del cepo cambiario y las modificaciones en las retenciones—, resulta fundamental contar con herramientas que permitan anticiparse a distintos escenarios.
        </div>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        '''
        <div style="text-align: justify;">
            Nuestro objetivo es crear una plataforma de simulación basada en modelos predictivos que ayude a los productores a trazar estrategias de largo plazo, optimizar sus decisiones y minimizar riesgos y pérdidas frente a un entorno cada vez más dinámico.
        </div>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        '''
        <div style="text-align: justify;">
            En esta primera etapa, desarrollaremos una versión Alpha de los algoritmos utilizando los datos actualmente disponibles. El objetivo inicial será predecir el precio de la leche en función de diversas variables económicas, como el índice de precios al consumidor (IPC), la cotización del dólar, el precio de determinados tipos de queso, entre otros factores relevantes.
        </div>
        ''',
        unsafe_allow_html=True
    )

    st.header("Datos disponibles")
    st.dataframe(df)

    st.header("Gráficos Ilustrativos:")

    # Mostrar las imágenes en columnas (una al lado de la otra)
    col1, col2 = st.columns(2)

    with col1:
        st.image(os.path.join(imagenes_dir, "mapa.png"), caption="Mapa de Unidades Productivas", use_container_width=True)

    with col2:
        st.image(os.path.join(imagenes_dir, "grafico.png"), caption="Gráfico de Barras", use_container_width=True)

    st.markdown(
        '''
        <div style="text-align: justify;">
           Según datos publicados por organismos oficiales como SENASA e INDEC, la provincia de Santa Fe concentra aproximadamente el 35% de la producción agropecuaria del país. Mejorar la eficiencia de este sector no solo permitiría aumentar su participación, sino también impulsar el desarrollo económico y social de toda la región.
        </div>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        '''
        <div style="text-align: justify;">
           En este contexto, consideramos fundamental trabajar en colaboración con el Centro de Economía Aplicada de la UNRAF para perfeccionar los algoritmos existentes y desarrollar nuevos modelos predictivos. Nuestro objetivo inicial es mejorar las condiciones de trabajo de los productores agropecuarios de nuestra región, promoviendo prácticas más eficientes y sostenibles.
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Variables específicas
    x = df['Mes']
    y = df['EXPORTACIONES             toneladas/mes']

    # Ordenar por Mes si es necesario (opcional)
    # df = df.sort_values('Mes')  

    # Calcular promedio móvil (suavizado)
    smoothed_y = uniform_filter1d(y, size=3)  # Puedes ajustar el tamaño de la ventana

    # Crear figura
    fig = go.Figure()

    # Barras
    fig.add_trace(go.Bar(
        x=x,
        y=y,
        name="Exportaciones",
        marker_color='blue',
        width=0.6  # Opcional: controla el ancho de barras
    ))

    # Línea suavizada
    fig.add_trace(go.Scatter(
        x=x,
        y=smoothed_y,
        mode='lines',
        name='Tendencia (Promedio Móvil)',
        line=dict(color='red', width=2)
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text="Exportaciones por Mes",
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(title="Mes"),
        yaxis=dict(title="Toneladas Exportadas"),
        bargap=0.2,
        template="plotly_white"
    )

    # Mostrar en Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '''
        <div style="text-align: justify;">
           Por otro lado, el análisis de los datos de exportaciones nos permite identificar cómo la estacionalidad influye de manera significativa en los volúmenes exportados. Esta observación abre la posibilidad de implementar algoritmos orientados a mitigar los efectos de dicha estacionalidad o a mejorar la eficiencia en los períodos de alta producción, con el fin de potenciar los picos de exportación y fortalecer la competitividad del sector a lo largo del año.
        </div>
        ''',
        unsafe_allow_html=True
    )

    st.markdown("""
    ## **Consideraciones**:
    
    - Para implementar algoritmos más complejos es necesario contar con más de 1000 registros para que el entrenamiento de los modelos sea más robusto.
    - La búsqueda de datos es engorrosa, ya que existen muchas fuentes diferentes, incompletas o incongruentes, lo que dificulta los procesos de estandarización de datos, proceso fundamental para un correcto entrenamiento de los modelos.
    - Las versiones Alpha presentadas utilizan únicamente regresión lineal.
    - Estas versiones fueron entrenadas con solo 100 registros, por lo que existe un amplio margen de mejora.

    
    """)

    st.markdown("### Funcionamiento del IPC en los datos utilizados")

    st.markdown("""
    El **Índice de Precios al Consumidor (IPC)** en esta aplicación se utiliza como un factor de ajuste para mantener los valores monetarios actualizados en términos reales. La serie histórica del IPC comienza con un valor base de **98** y se va modificando mensualmente según los valores oficiales del **IMPCe**, generando una serie acumulativa que refleja la evolución de los precios a lo largo del tiempo.

    A la hora de ingresar manualmente un nuevo dato de IPC (por ejemplo, para proyectar meses futuros), es necesario partir del **último valor disponible**, que actualmente es **7864,1257** correspondiente a **enero de 2025**.
    """)

    st.markdown("Ejemplo de cálculo para proyectar IPC de abril de 2025:")

    st.markdown("""
    - IPC de **febrero 2025**: se incrementó un **2,1%**, lo que equivale a multiplicar por **1,021**
    - IPC de **marzo 2025**: se incrementó un **3,7%**, lo que equivale a multiplicar por **1,037**
    """)

    st.code("IPC abril 2025 = 7864,1257 × 1,021 × 1,037", language="python")

    st.markdown("""
    De esta manera, se mantiene la coherencia con la lógica acumulativa del índice desde su base inicial, permitiendo que los modelos realicen comparaciones históricas y proyecciones de manera consistente.
    """)


            
