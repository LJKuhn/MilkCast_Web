def C_visualizacion(df, df2):
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

    st.header("Datos Iniciales:")
    # Mostrar datos más recientes primero - ordenados por índice descendente
    df_sorted = df.iloc[::-1].reset_index(drop=True)
    
    # Configurar altura fija para mantener consistencia visual
    st.dataframe(df_sorted, height=400, use_container_width=True)

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

    # Gráfico de Evolución de Rentabilidad
    st.subheader("📈 Evolución de la Rentabilidad")
    
    st.markdown(
        '''
        <div style="text-align: justify;">
            La rentabilidad es un indicador clave que refleja la diferencia entre los ingresos obtenidos por la venta de leche y los costos de producción. 
            Este gráfico muestra la evolución temporal de la rentabilidad del sector lácteo, permitiendo identificar períodos de mayor y menor 
            rentabilidad, así como tendencias de largo plazo. La línea roja punteada marca el punto de equilibrio (rentabilidad = 0), donde los 
            ingresos igualan a los costos. Los períodos por encima de esta línea indican rentabilidad positiva, mientras que los períodos por 
            debajo muestran pérdidas operativas.
        </div>
        ''',
        unsafe_allow_html=True
    )
    
    # Verificar si existe la columna de rentabilidad en alguno de los datasets
    rentabilidad_col = None
    dataset_usado = None
    
    # Buscar en df2 primero (datos ampliados)
    if 'RENTABILIDAD' in df2.columns:
        rentabilidad_col = 'RENTABILIDAD'
        dataset_usado = df2
        mes_col = 'Mes' if 'Mes' in df2.columns else df2.index
    # Si no está en df2, buscar en df
    elif 'RENTABILIDAD' in df.columns:
        rentabilidad_col = 'RENTABILIDAD'
        dataset_usado = df
        mes_col = 'Mes' if 'Mes' in df.columns else df.index
    
    if rentabilidad_col and dataset_usado is not None:
        # Crear el gráfico de rentabilidad
        fig_rentabilidad = go.Figure()
        
        # Línea principal de rentabilidad
        fig_rentabilidad.add_trace(go.Scatter(
            x=dataset_usado[mes_col] if isinstance(mes_col, str) else mes_col,
            y=dataset_usado[rentabilidad_col],
            mode='lines+markers',
            name='Rentabilidad',
            line=dict(color='#2E8B57', width=3),
            marker=dict(size=6, color='#2E8B57'),
            hovertemplate='<b>Mes:</b> %{x}<br><b>Rentabilidad:</b> %{y:.4f}<extra></extra>'
        ))
        
        # Línea de referencia en 0 (punto de equilibrio)
        fig_rentabilidad.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="red", 
            annotation_text="Punto de Equilibrio",
            annotation_position="bottom right"
        )
        
        # Calcular promedio móvil para suavizar
        try:
            rentabilidad_suavizada = uniform_filter1d(dataset_usado[rentabilidad_col], size=3)
            fig_rentabilidad.add_trace(go.Scatter(
                x=dataset_usado[mes_col] if isinstance(mes_col, str) else mes_col,
                y=rentabilidad_suavizada,
                mode='lines',
                name='Tendencia (Promedio Móvil)',
                line=dict(color='#FF6B35', width=2, dash='dot'),
                opacity=0.8
            ))
        except:
            pass  # Si no se puede calcular el promedio móvil, continuar sin él
        
        # Configurar layout
        fig_rentabilidad.update_layout(
            title=dict(
                text="Evolución de la Rentabilidad del Sector Lácteo",
                x=0.5,
                xanchor='center',
                font=dict(size=16, color='#2F4F4F')
            ),
            xaxis=dict(
                title="Período",
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            yaxis=dict(
                title="Rentabilidad",
                gridcolor='lightgray',
                gridwidth=0.5,
                zeroline=True,
                zerolinecolor='red',
                zerolinewidth=1
            ),
            template="plotly_white",
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Colorear el área según rentabilidad positiva/negativa
        rentabilidad_values = dataset_usado[rentabilidad_col].values
        for i in range(len(rentabilidad_values)):
            color = 'rgba(46, 139, 87, 0.1)' if rentabilidad_values[i] >= 0 else 'rgba(220, 20, 60, 0.1)'
            fig_rentabilidad.add_shape(
                type="rect",
                x0=i-0.4, x1=i+0.4,
                y0=0, y1=rentabilidad_values[i],
                fillcolor=color,
                line=dict(width=0),
                layer="below"
            )
        
        st.plotly_chart(fig_rentabilidad, use_container_width=True)
        
        # Análisis estadístico de la rentabilidad
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        
        rentabilidad_actual = dataset_usado[rentabilidad_col].iloc[-1]
        rentabilidad_promedio = dataset_usado[rentabilidad_col].mean()
        rentabilidad_max = dataset_usado[rentabilidad_col].max()
        rentabilidad_min = dataset_usado[rentabilidad_col].min()
        meses_positivos = (dataset_usado[rentabilidad_col] > 0).sum()
        total_meses = len(dataset_usado[rentabilidad_col])
        
        with col_stats1:
            st.metric(
                "Rentabilidad Actual", 
                f"{rentabilidad_actual:.4f}",
                f"{((rentabilidad_actual/rentabilidad_promedio-1)*100):+.1f}% vs promedio" if rentabilidad_promedio != 0 else None
            )
        
        with col_stats2:
            st.metric(
                "Rentabilidad Promedio", 
                f"{rentabilidad_promedio:.4f}",
                "📊 Histórico"
            )
        
        with col_stats3:
            st.metric(
                "Rango", 
                f"{rentabilidad_min:.4f} - {rentabilidad_max:.4f}",
                f"Amplitud: {(rentabilidad_max - rentabilidad_min):.4f}"
            )
        
        with col_stats4:
            porcentaje_positivo = (meses_positivos / total_meses) * 100
            st.metric(
                "Meses Rentables", 
                f"{meses_positivos}/{total_meses}",
                f"{porcentaje_positivo:.1f}% del tiempo"
            )
        
        # Interpretación
        st.markdown("### 💡 Interpretación de la Rentabilidad")
        
        if rentabilidad_actual > 0:
            if rentabilidad_actual > rentabilidad_promedio:
                st.success(f"🟢 **Situación Favorable:** La rentabilidad actual ({rentabilidad_actual:.4f}) está por encima del promedio histórico ({rentabilidad_promedio:.4f}). El sector muestra un desempeño sólido.")
            else:
                st.info(f"🔵 **Situación Estable:** La rentabilidad actual ({rentabilidad_actual:.4f}) es positiva pero por debajo del promedio histórico. Hay margen de mejora.")
        else:
            st.warning(f"🟡 **Situación Desafiante:** La rentabilidad actual ({rentabilidad_actual:.4f}) es negativa. Es importante analizar los factores que afectan la rentabilidad del sector.")
        
        # Recomendaciones según el estado
        if porcentaje_positivo >= 70:
            st.info("📈 **Tendencia General:** El sector muestra rentabilidad positiva en la mayoría del período analizado, indicando un negocio generalmente viable.")
        elif porcentaje_positivo >= 50:
            st.warning("⚖️ **Volatilidad Moderada:** La rentabilidad fluctúa considerablemente. Es importante considerar estrategias de gestión de riesgos.")
        else:
            st.error("📉 **Alta Volatilidad:** El sector presenta desafíos de rentabilidad frecuentes. Se recomienda un análisis profundo de costos y eficiencias.")
    
    else:
        st.info("ℹ️ Los datos de rentabilidad no están disponibles en el dataset actual. Este gráfico se mostrará cuando los datos incluyan la variable 'RENTABILIDAD'.")
        
        # Mostrar un gráfico placeholder o alternativo si hay otras variables interesantes
        if 'Precio/litro Nacional - SIGLeA' in df.columns and 'COSTO' in df.columns:
            st.write("**Análisis alternativo: Relación Precio vs Costo**")
            
            fig_alt = go.Figure()
            fig_alt.add_trace(go.Scatter(
                x=df['Mes'],
                y=df['Precio/litro Nacional - SIGLeA'],
                mode='lines',
                name='Precio por Litro',
                line=dict(color='green')
            ))
            
            if 'COSTO' in df.columns:
                fig_alt.add_trace(go.Scatter(
                    x=df['Mes'],
                    y=df['COSTO'],
                    mode='lines',
                    name='Costo',
                    line=dict(color='red')
                ))
            
            fig_alt.update_layout(
                title="Evolución de Precios y Costos",
                xaxis_title="Mes",
                yaxis_title="Valor ($)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_alt, use_container_width=True)

    # Gráfico adicional: Análisis de Variables Macroeconómicas
    st.subheader("💰 Impacto de Variables Macroeconómicas")
    
    st.markdown(
        '''
        <div style="text-align: justify;">
            Las variables macroeconómicas como el Índice de Precios al Consumidor (IPC) y el tipo de cambio del dólar oficial tienen un 
            impacto directo en el sector lácteo. El IPC refleja la inflación general de la economía, afectando tanto los costos de producción 
            como los precios de venta. El dólar oficial influye en la competitividad de las exportaciones lácteas y en el costo de insumos 
            importados. Este gráfico permite visualizar la evolución conjunta de estas variables clave para entender su relación con la 
            dinámica del sector.
        </div>
        ''',
        unsafe_allow_html=True
    )
    
    # Verificar disponibilidad de variables macroeconómicas
    variables_macro_disponibles = []
    if 'IPC-Mensual' in df2.columns:
        variables_macro_disponibles.append(('IPC-Mensual', df2, 'IPC Mensual', '#1f77b4'))
    elif 'IPC-Mensual' in df.columns:
        variables_macro_disponibles.append(('IPC-Mensual', df, 'IPC Mensual', '#1f77b4'))
    
    if 'DOLAR OFICIAL $/US$' in df2.columns:
        variables_macro_disponibles.append(('DOLAR OFICIAL $/US$', df2, 'Dólar Oficial ($/US$)', '#ff7f0e'))
    elif 'DOLAR OFICIAL $/US$' in df.columns:
        variables_macro_disponibles.append(('DOLAR OFICIAL $/US$', df, 'Dólar Oficial ($/US$)', '#ff7f0e'))
    
    if 'Precio/litro Nacional - SIGLeA' in df2.columns:
        variables_macro_disponibles.append(('Precio/litro Nacional - SIGLeA', df2, 'Precio Leche ($/L)', '#2ca02c'))
    elif 'Precio/litro Nacional - SIGLeA' in df.columns:
        variables_macro_disponibles.append(('Precio/litro Nacional - SIGLeA', df, 'Precio Leche ($/L)', '#2ca02c'))
    
    if len(variables_macro_disponibles) >= 2:
        # Crear el gráfico con múltiples ejes Y
        fig_macro = go.Figure()
        
        # Configurar colores y ejes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        y_axes = ['y', 'y2', 'y3', 'y4']
        
        for i, (col_name, dataset, display_name, color) in enumerate(variables_macro_disponibles[:4]):
            mes_col = 'Mes' if 'Mes' in dataset.columns else dataset.index
            
            fig_macro.add_trace(go.Scatter(
                x=dataset[mes_col] if isinstance(mes_col, str) else mes_col,
                y=dataset[col_name],
                mode='lines+markers',
                name=display_name,
                line=dict(color=color, width=2.5),
                marker=dict(size=4),
                yaxis=y_axes[i] if i < len(y_axes) else 'y',
                hovertemplate=f'<b>{display_name}:</b> %{{y:.2f}}<br><b>Período:</b> %{{x}}<extra></extra>'
            ))
        
        # Configurar layout con múltiples ejes Y
        layout_config = {
            'title': dict(
                text="Evolución de Variables Macroeconómicas Clave",
                x=0.5,
                xanchor='center',
                font=dict(size=16, color='#2F4F4F')
            ),
            'xaxis': dict(
                title="Período",
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            'template': "plotly_white",
            'height': 500,
            'hovermode': 'x unified',
            'legend': dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        }
        
        # Configurar ejes Y según el número de variables
        if len(variables_macro_disponibles) >= 1:
            layout_config['yaxis'] = dict(
                title=variables_macro_disponibles[0][2],
                side="left",
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=0.5
            )
        
        if len(variables_macro_disponibles) >= 2:
            layout_config['yaxis2'] = dict(
                title=variables_macro_disponibles[1][2],
                side="right",
                overlaying="y",
                showgrid=False
            )
        
        if len(variables_macro_disponibles) >= 3:
            layout_config['yaxis3'] = dict(
                title=variables_macro_disponibles[2][2],
                side="left",
                overlaying="y",
                position=0.05,
                showgrid=False
            )
        
        fig_macro.update_layout(**layout_config)
        
        st.plotly_chart(fig_macro, use_container_width=True)
        
        # Análisis estadístico de correlaciones simples
        st.markdown("### 🔍 Análisis de Relaciones")
        
        col_anal1, col_anal2 = st.columns(2)
        
        with col_anal1:
            # Calcular algunas correlaciones básicas si hay suficientes variables
            if len(variables_macro_disponibles) >= 2:
                var1_name, var1_dataset, var1_display, _ = variables_macro_disponibles[0]
                var2_name, var2_dataset, var2_display, _ = variables_macro_disponibles[1]
                
                # Verificar que ambas variables estén en el mismo dataset o crear uno combinado
                if var1_dataset.equals(var2_dataset):
                    corr_value = var1_dataset[var1_name].corr(var2_dataset[var2_name])
                    st.metric(
                        f"Correlación: {var1_display} vs {var2_display}",
                        f"{corr_value:.3f}",
                        "Correlación lineal"
                    )
                    
                    if abs(corr_value) > 0.7:
                        st.success("🟢 Correlación fuerte detectada")
                    elif abs(corr_value) > 0.3:
                        st.info("🔵 Correlación moderada")
                    else:
                        st.warning("🟡 Correlación débil")
        
        with col_anal2:
            # Mostrar estadísticas del período más reciente
            if variables_macro_disponibles:
                var_principal = variables_macro_disponibles[0]
                valor_actual = var_principal[1][var_principal[0]].iloc[-1]
                valor_promedio = var_principal[1][var_principal[0]].mean()
                
                st.metric(
                    f"{var_principal[2]} Actual",
                    f"{valor_actual:.2f}",
                    f"{((valor_actual/valor_promedio-1)*100):+.1f}% vs promedio"
                )
        
        # Interpretación económica
        st.markdown("### 💡 Interpretación Económica")
        
        interpretaciones = []
        
        # Buscar patrones específicos
        ipc_data = next((item for item in variables_macro_disponibles if 'IPC' in item[0]), None)
        dolar_data = next((item for item in variables_macro_disponibles if 'DOLAR' in item[0]), None)
        precio_data = next((item for item in variables_macro_disponibles if 'Precio' in item[0]), None)
        
        if ipc_data and dolar_data:
            ipc_actual = ipc_data[1][ipc_data[0]].iloc[-1]
            ipc_anterior = ipc_data[1][ipc_data[0]].iloc[-2] if len(ipc_data[1]) > 1 else ipc_actual
            dolar_actual = dolar_data[1][dolar_data[0]].iloc[-1]
            
            if ipc_actual > ipc_anterior * 1.02:  # Más del 2% de aumento
                interpretaciones.append("📈 **Presión inflacionaria:** El IPC muestra aumentos significativos que pueden impactar en los costos de producción.")
            
            if dolar_actual > 800:  # Umbral ejemplo
                interpretaciones.append("💱 **Tipo de cambio elevado:** El dólar alto puede favorecer las exportaciones lácteas pero encarecer insumos importados.")
        
        if precio_data:
            precio_actual = precio_data[1][precio_data[0]].iloc[-1]
            precio_promedio = precio_data[1][precio_data[0]].mean()
            
            if precio_actual > precio_promedio * 1.1:
                interpretaciones.append("🥛 **Precios favorables:** El precio actual de la leche está por encima del promedio histórico.")
            elif precio_actual < precio_promedio * 0.9:
                interpretaciones.append("🥛 **Presión en precios:** El precio actual está por debajo del promedio histórico.")
        
        if interpretaciones:
            for interpretacion in interpretaciones:
                st.info(interpretacion)
        else:
            st.info("📊 **Análisis continuo:** Las variables macroeconómicas requieren monitoreo constante para identificar tendencias y oportunidades en el sector lácteo.")
    
    else:
        st.info("ℹ️ Las variables macroeconómicas principales (IPC, Dólar, Precio de Leche) no están completamente disponibles en el dataset actual. Este análisis se mostrará cuando los datos incluyan estas variables clave.")

    st.header("Datos Ampliados:")
    # Mostrar datos más recientes primero - ordenados por índice descendente
    df2_sorted = df2.iloc[::-1].reset_index(drop=True)
    
    # Configurar altura fija para mantener consistencia visual
    st.dataframe(df2_sorted, height=400, use_container_width=True)

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

    st.code("IPC finales de marzo/inicio de abril 2025 = 7864,1257 × 1,021 × 1,037 =  8326,3554", language="python")

    st.markdown("""
    De esta manera, se mantiene la coherencia con la lógica acumulativa del índice desde su base inicial, permitiendo que los modelos realicen comparaciones históricas y proyecciones de manera consistente.
    """)


            
