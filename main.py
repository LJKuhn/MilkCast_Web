# Librerías
import streamlit as st
import pandas as pd
import pickle
import os
import gdown

# Componentes
from componentes.componente_prediccion import C_prediccion
from componentes.componente_clasificacion import C_clasificacion
from componentes.componente_eda import C_visualizacion

# Configuración de la página
st.set_page_config(page_title="Ayuda al tambo", layout="wide")

# Usar HTML para personalizar el tamaño del texto
st.markdown("""
    <style>
        .custom-font {
            font-family: 'Georgia', serif;
            font-size: 100px;  /* Tamaño de fuente 100px */
            color: #FF6347;  /* Cambia el color a tu gusto */
        }
    </style>
    <h1 class="custom-font">MilkCast</h1>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size: 50px;'>ML en el sector agropecuario</h1>", unsafe_allow_html=True)

# Creación de pestañas
tab1, tab2, tab3 = st.tabs(["Datos y Gráficos", "Prediccion con IPC y dolar", "Prediccion con productos"])

# URLs de los modelos en Google Drive (usando el formato adecuado para gdown)
url_modelprediccionUS = "https://drive.google.com/file/d/1-sjXKNng7Mxevubem3ei54m0cYuv132D"
url_modelprediccionPr = "https://drive.google.com/file/d/1mElfQCidhvGXT86gPnoKEaXPRtrWqVdJ"

# Directorio donde se guardarán los modelos
modelos_dir = "modelos"
os.makedirs(modelos_dir, exist_ok=True)

# Función para descargar un archivo si no existe
def descargar_modelo(url, path):
    if not os.path.exists(path):
        print(f"Descargando {os.path.basename(path)}...")
        try:
            # Utilizamos gdown para descargar el archivo
            gdown.download(url, path, quiet=False)
            print(f"Archivo descargado y guardado en {path}.")
        except Exception as e:
            print(f"Error en la descarga: {e}")
    else:
        print(f"El archivo {os.path.basename(path)} ya existe.")

# Descargar los modelos solo si no existen
descargar_modelo(url_modelprediccionUS, os.path.join(modelos_dir, "modelo_regresion-Precio-IPC-Dolar.pkl"))
descargar_modelo(url_modelprediccionPr, os.path.join(modelos_dir, "modelo_regresion-Precio-ComEnt-Queso-Yogur.pkl"))

# Función para cargar los modelos en caché usando st.cache_resource
@st.cache_resource
def cargar_modelo(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

# Cargar los modelos (se cargan solo una vez)
model1 = cargar_modelo(os.path.join(modelos_dir, "modelo_regresion-Precio-IPC-Dolar.pkl"))
model2 = cargar_modelo(os.path.join(modelos_dir, "modelo_regresion-Precio-ComEnt-Queso-Yogur.pkl"))

# Función para mostrar tipo y verificar método 'predict' en los modelos
def mostrar_info_modelo(model, nombre_modelo):
    tipo_modelo = type(model)
    tiene_predict = hasattr(model, 'predict')
    if tiene_predict:
        st.markdown(f"**{nombre_modelo}:** El modelo es de tipo `{tipo_modelo}` y tiene el método `predict`.")
    else:
        st.markdown(f"**{nombre_modelo}:** El modelo es de tipo `{tipo_modelo}` y **no tiene** el método `predict`.")

# Mostrar la información sobre los modelos cargados
with st.expander("Información de los Modelos"):
    mostrar_info_modelo(model2, "Modelo de Predicción")
    mostrar_info_modelo(model1, "Modelo de Clasificación")

# URLs de las imágenes en Google Drive (en formato adecuado)
url_mapa_unidades = "https://drive.google.com/file/d/1PgVHUgz2u9iOgDUcF4UL4JgVc9DUJj4u"  
url_grafico_barras = "https://drive.google.com/file/d/1Ht0_HTgXLnOLwoPUFL6AMyakgz40S2tq"  

# Directorio donde se guardarán las imágenes
imagenes_dir = "imagenes"
os.makedirs(imagenes_dir, exist_ok=True)

# Función para descargar una imagen si no existe
def descargar_imagen(url, path):
    if not os.path.exists(path):
        print(f"Descargando {os.path.basename(path)}...")
        try:
            # Utilizamos gdown para descargar la imagen
            gdown.download(url, path, quiet=False)
            print(f"Imagen descargada y guardada en {path}.")
        except Exception as e:
            print(f"Error en la descarga de la imagen: {e}")
    else:
        print(f"La imagen {os.path.basename(path)} ya existe.")

# Descargar las imágenes si no existen
descargar_imagen(url_mapa_unidades, os.path.join(imagenes_dir, "mapa.png"))
descargar_imagen(url_grafico_barras, os.path.join(imagenes_dir, "grafico.png"))


# Cargar el archivo combinado (usado para las visualizaciones)
df = pd.read_csv("modelos/archivo.csv")

# Función para limpiar el estado cuando se cambia de tab
def limpiar_estado_tab_actual(tab_seleccionado):
    if tab_seleccionado != "Datos y Gráficos" and 'input_data_eda' in st.session_state:
        del st.session_state['input_data_eda']
    if tab_seleccionado != "Prediccion con IPC y dolar" and 'input_data_prediccionUs' in st.session_state:
        del st.session_state['input_data_prediccionUs']
    if tab_seleccionado != "Prediccion con productos" and 'input_data_prediccionPr' in st.session_state:
        del st.session_state['input_data_prediccionPr']

# Contenido de la pestaña 1: Datos y Gráficos
with tab1:
    limpiar_estado_tab_actual("Datos y Gráficos")  # Limpiar las otras pestañas al entrar a esta
    C_visualizacion(df)

    st.markdown(
        '''
        <div style="text-align: justify;">
           # Links de informacion
        ### Modelos en Google Drive
        - url_modelprediccionUs = "https://drive.google.com/file/d/1-sjXKNng7Mxevubem3ei54m0cYuv132D/view?usp=sharing"
        - url_modelprediccionPr = "https://drive.google.com/file/d/1mElfQCidhvGXT86gPnoKEaXPRtrWqVdJ/view?usp=sharing"

        ### Datos producción lechera
        - https://www.magyp.gob.ar/sitio/areas/ss_lecheria/ 
        - https://www.indec.gob.ar/indec/web/Nivel4-Tema-3-8-89 
        - https://www.ocla.org.ar/ 
        - https://datos.gob.ar/ar/dataset?tags=leche&groups=agri 
        
        ### Datos climáticos
        - https://www.argentina.gob.ar/informacion-agroclimatica 
        - https://www.argentina.gob.ar/inta 
        - https://siga.inta.gob.ar/#/data 
        
        ### Informe o muestreo ejemplos
        - https://www.unraf.edu.ar/index.php/menucontenidos/164-centro-de-enconomia-aplicada/2962-cea 
        
        ### Informe de agronomía hecho por la UNRaf
        - https://app.powerbi.com/view?r=eyJrIjoiYjQyZjZhZTktOGEwYi00MjdmLWIzZWMtYTFkNTcxN2EzNTQzIiwidCI6IjFiYWViYmNhLWJjNjYtNDc4My05OWJiLTI5OGIxYzNmMTA4MyJ9 
        
        ### Informe de unidades productivas por provincias
        - https://www.ocla.org.ar/noticias/29882853-unidades-productivas-y-rodeo-lechero-a-marzo-de-2024 
        - https://www.argentina.gob.ar/sites/default/files/87-caracterizacion_tambos_bovinos_diciembre_2021.pdf 
        - https://www.infortambo.com/blog/en-el-2024-el-numero-de-tambos-cayo-a-9-735-unidades-productivas-de-leche/
        </div>
        ''',
        unsafe_allow_html=True
    )

# Contenido de la pestaña 2: Prediccion con IPC y dolar
with tab2:
    limpiar_estado_tab_actual("Prediccion con IPC y dolar")  # Limpiar las otras pestañas al entrar a esta
    C_prediccion(model1)

# Contenido de la pestaña 3: Prediccion con productos
with tab3:
    limpiar_estado_tab_actual("Prediccion con productos")  # Limpiar las otras pestañas al entrar a esta
    C_clasificacion(model2)
