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
# Nuevos componentes OCLA
from componentes.componente_rentabilidad import C_rentabilidad
from componentes.componente_costos import C_costos
from componentes.componente_precio_queso import C_precio_queso
from componentes.componente_precio_internacional import C_precio_internacional
from componentes.componente_precio_novillos import C_precio_novillos
from componentes.componente_variables_macro import C_variables_macro
from componentes.componente_productos_lacteos import C_productos_lacteos

# Configuración de la página
st.set_page_config(page_title="MilkCast", layout="wide")

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Datos y Gráficos", 
    "💱 IPC y Dólar", 
    "🥛 Productos Básicos",
    "🎯 Rentabilidad", 
    "💰 Costos", 
    "🧀 Precio Queso",
    "🌍 Internacional", 
    "🐄 Novillos", 
    "📊 Variables Macro", 
    "🥛 Productos H"
])

# URLs de los modelos y archivos en Google Drive (usando el formato adecuado para gdown)
# Aquí se centralizan todas las descargas que necesita la app.
# Reemplaza los valores "REEMPLAZAR_ID_..." por los IDs reales de Drive (o las URLs completas).
files_to_download = {
    # Modelos ya usados en la app
    "https://drive.google.com/file/d/1lXK5IvmoBYuFEDO35O6yv9yAjITrNUXh": "modelo_regresion-Precio-IPC-Dolar.pkl",
    "https://drive.google.com/file/d/1WKw9ZazXYoeppAyE_NT-iAkFWkuum8mN": "modelo_regresion-Precio-ComEnt-Queso-Yogur.pkl",

    # Modelos OCLA (REEMPLAZAR con los IDs/URLs reales)
    "https://drive.google.com/file/d/1yvkKHlS2sHxZ6uvIafQqORn4hE6cQl3s": "modelo_A_rentabilidad.pkl",
    "https://drive.google.com/file/d/12wjhEPeJ8I08bVJwIEHTnbudR737P7St": "modelo_B_costos.pkl",
    "https://drive.google.com/file/d/1LLBwmvRzB4TXC4UX8bYPpoRVTMy0bifh": "modelo_D_precio_queso.pkl",
    "https://drive.google.com/file/d/1kDDitTJ4HpiABZHqR7a8uLBF75Lr1YKP": "modelo_E_precio_internacional.pkl",
    "https://drive.google.com/file/d/1E1f9LPMFY6RU_G4etqkdANCXy1BpOE-q": "modelo_F_precio_novillos.pkl",
    "https://drive.google.com/file/d/1aNZIMH9_MOeAJrmTBrT7T4rGHLI8e4Wg": "modelo_G_variables_macroeconomicas.pkl",
    "https://drive.google.com/file/d/1e-dMgzQDdzUM10RFojdwkAzpIiLZgkm8": "modelo_H_productos_lacteos.pkl",

    # CSVs usados por la app (REEMPLAZAR con IDs/URLs reales)
    "https://drive.google.com/file/d/1-_Co4-GszOdutP0Z5tAEWTmjDqKEuEMG": "archivo.csv",
    "https://drive.google.com/file/d/1hdF61HgxE_ESN8cOYM3D3GBehxpyCyaG": "dataset_LIMPIO_original.csv",
}

# Directorio donde se guardarán los modelos
modelos_dir = "modelos"
os.makedirs(modelos_dir, exist_ok=True)

# Función para descargar un archivo si no existe
def descargar_modelo(url, path):
    if not os.path.exists(path):
        print(f"Descargando {os.path.basename(path)}...")
        try:
            # Utilizamos gdown para descargar el archivo
            # gdown acepta tanto URLs tipo /file/d/ID como /uc?id=ID
            gdown.download(url, path, quiet=False)
            print(f"Archivo descargado y guardado en {path}.")
        except Exception as e:
            print(f"Error en la descarga: {e}")
    else:
        print(f"El archivo {os.path.basename(path)} ya existe.")

# Descargar todos los archivos listados en files_to_download (si no existen)
for url, filename in files_to_download.items():
    target_path = os.path.join(modelos_dir, filename)
    descargar_modelo(url, target_path)

# Función para cargar los modelos en caché usando st.cache_resource
@st.cache_resource
def cargar_modelo(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

# Función para cargar modelos locales (DESDE CARPETA MODELOS)
@st.cache_resource
def cargar_modelo_local(nombre_modelo):
    """Carga modelos desde la carpeta modelos de MilkCast-V2"""
    try:
        # Ruta hacia la carpeta modelos dentro de MilkCast-V2
        ruta = os.path.join(modelos_dir, f"{nombre_modelo}.pkl")
        if os.path.exists(ruta):
            with open(ruta, 'rb') as f:
                return pickle.load(f)
        else:
            st.error(f"❌ No se encontró el modelo: {ruta}")
            return None
    except Exception as e:
        st.error(f"❌ Error cargando {nombre_modelo}: {str(e)}")
        return None

# Cargar los modelos existentes (se cargan solo una vez)
model1 = cargar_modelo(os.path.join(modelos_dir, "modelo_regresion-Precio-IPC-Dolar.pkl"))
model2 = cargar_modelo(os.path.join(modelos_dir, "modelo_regresion-Precio-ComEnt-Queso-Yogur.pkl"))

# Cargar los nuevos modelos OCLA (PARA DESARROLLO LOCAL)
try:
    model_A = cargar_modelo_local("modelo_A_rentabilidad")
    model_B = cargar_modelo_local("modelo_B_costos") 
    model_D = cargar_modelo_local("modelo_D_precio_queso")
    model_E = cargar_modelo_local("modelo_E_precio_internacional")
    model_F = cargar_modelo_local("modelo_F_precio_novillos")
    model_G = cargar_modelo_local("modelo_G_variables_macroeconomicas") 
    model_H = cargar_modelo_local("modelo_H_productos_lacteos")
    
    modelos_ocla_cargados = True
    st.success("✅ Todos los modelos OCLA cargados correctamente")
    
except Exception as e:
    modelos_ocla_cargados = False
    st.warning(f"⚠️ Algunos modelos OCLA no pudieron cargarse: {e}")
    
    # Crear modelos dummy para evitar errores
    model_A = model_B = model_D = model_E = model_F = model_G = model_H = None

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
url_mapa_unidades = "https://drive.google.com/uc?id=1PgVHUgz2u9iOgDUcF4UL4JgVc9DUJj4u"
url_grafico_barras = "https://drive.google.com/uc?id=1Ht0_HTgXLnOLwoPUFL6AMyakgz40S2tq"

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
df2 = pd.read_csv("modelos/dataset_LIMPIO_original.csv")

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

    C_visualizacion(df, df2)

    st.markdown(
        '''
        <div style="text-align: justify;">
           
        # Links de informacion

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

# Contenido de la pestaña 4: Rentabilidad (Modelo A)
with tab4:
    limpiar_estado_tab_actual("Rentabilidad")
    if modelos_ocla_cargados and model_A is not None:
        C_rentabilidad(model_A)
    else:
        st.error("❌ Modelo A (Rentabilidad) no disponible. Verifique que el archivo PKL esté en la carpeta correcta.")
        st.info("📁 Ruta esperada: modelos/modelo_A_rentabilidad.pkl")

# Contenido de la pestaña 5: Costos (Modelo B)
with tab5:
    limpiar_estado_tab_actual("Costos")
    if modelos_ocla_cargados and model_B is not None:
        C_costos(model_B)
    else:
        st.error("❌ Modelo B (Costos) no disponible. Verifique que el archivo PKL esté en la carpeta correcta.")
        st.info("📁 Ruta esperada: modelos/modelo_B_costos.pkl")

# Contenido de la pestaña 6: Precio Queso (Modelo D)
with tab6:
    limpiar_estado_tab_actual("Precio Queso")
    if modelos_ocla_cargados and model_D is not None:
        C_precio_queso(model_D)
    else:
        st.error("❌ Modelo D (Precio Queso) no disponible. Verifique que el archivo PKL esté en la carpeta correcta.")
        st.info("📁 Ruta esperada: modelos/modelo_D_precio_queso.pkl")

# Contenido de la pestaña 7: Precio Internacional (Modelo E)
with tab7:
    limpiar_estado_tab_actual("Internacional")
    if modelos_ocla_cargados and model_E is not None:
        C_precio_internacional(model_E)
    else:
        st.error("❌ Modelo E (Precio Internacional) no disponible. Verifique que el archivo PKL esté en la carpeta correcta.")
        st.info("📁 Ruta esperada: modelos/modelo_E_precio_internacional.pkl")

# Contenido de la pestaña 8: Precio Novillos (Modelo F)
with tab8:
    limpiar_estado_tab_actual("Novillos")
    if modelos_ocla_cargados and model_F is not None:
        C_precio_novillos(model_F)
    else:
        st.error("❌ Modelo F (Precio Novillos) no disponible. Verifique que el archivo PKL esté en la carpeta correcta.")
        st.info("📁 Ruta esperada: modelos/modelo_F_precio_novillos.pkl")

# Contenido de la pestaña 9: Variables Macro (Modelo G)
with tab9:
    limpiar_estado_tab_actual("Variables Macro")
    if modelos_ocla_cargados and model_G is not None:
        C_variables_macro(model_G)
    else:
        st.error("❌ Modelo G (Variables Macro) no disponible. Verifique que el archivo PKL esté en la carpeta correcta.")
        st.info("📁 Ruta esperada: modelos/modelo_G_variables_macroeconomicas.pkl")

# Contenido de la pestaña 10: Productos Lácteos (Modelo H)
with tab10:
    limpiar_estado_tab_actual("Productos H")
    if modelos_ocla_cargados and model_H is not None:
        C_productos_lacteos(model_H)
    else:
        st.error("❌ Modelo H (Productos Lácteos) no disponible. Verifique que el archivo PKL esté en la carpeta correcta.")
        st.info("📁 Ruta esperada: modelos/modelo_H_productos_lacteos.pkl")
