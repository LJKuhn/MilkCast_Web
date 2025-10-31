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
# IMPORTANTE: Para que las descargas funcionen correctamente:
# 1. Los archivos en Google Drive deben ser públicos (compartir con "cualquier persona con el enlace")
# 2. Usar el formato: https://drive.google.com/file/d/ID_DEL_ARCHIVO
# 3. Si hay errores de descarga, verificar permisos en Google Drive
files_to_download = {
    # Modelos ya usados en la app
    "https://drive.google.com/file/d/16ZqjF63HtFxV7fIVqJY-EQgIP0nGBgSA": "modelo_regresion-Precio-IPC-Dolar.pkl",
    "https://drive.google.com/file/d/1lp3kmeGeTPVx8-QZu6HFcSPqJFbU2zya": "modelo_regresion-Precio-ComEnt-Queso-Yogur.pkl",

    # Modelos OCLA (REEMPLAZAR con los IDs/URLs reales)
    "https://drive.google.com/file/d/1HVDxaJpuWk3rUIPOlXBsIFyWW32syuRi": "modelo_A_rentabilidad.pkl",
    "https://drive.google.com/file/d/1kRIzujwRxQzJ9b738D_oBPMJVNeU9DLZ": "modelo_B_costos.pkl",
    "https://drive.google.com/file/d/1r5PDqxLNQOD2QKQXJd5XFfKg5K72V_Jm": "modelo_D_precio_queso.pkl",
    "https://drive.google.com/file/d/1BjwGFe_djZ3c3W6XuedKQkorDB4cp4Yk": "modelo_E_precio_internacional.pkl",
    "https://drive.google.com/file/d/1VprLKVHthzzGnt7MB14KyTPmL9qUsGvt": "modelo_F_precio_novillos.pkl",
    "https://drive.google.com/file/d/1qcHUmGB9DKe9lrmzkfUZrvBPGQF7Sedh": "modelo_G_variables_macroeconomicas.pkl",
    "https://drive.google.com/file/d/1plMbfsdqBAZAJxy9ziQdflrsD39_Fi13": "modelo_H_productos_lacteos.pkl",

    # CSVs usados por la app (REEMPLAZAR con IDs/URLs reales)
    "https://drive.google.com/file/d/1oa0iGxqlGgOmWpfLWO7pMeYjGuv4AFzV": "archivo.csv",
    "https://drive.google.com/file/d/1Xr7IbFcIdZvbrCsqKzv7fR-XW8zhdZ2I": "dataset_LIMPIO_original.csv",
}

# Directorio donde se guardarán los modelos
modelos_dir = "modelos"
os.makedirs(modelos_dir, exist_ok=True)

# Función para verificar si un archivo PKL es válido
def verificar_archivo_pkl(path):
    """Verifica si un archivo PKL es válido y no está corrupto"""
    try:
        with open(path, 'rb') as f:
            # Leer los primeros bytes para verificar que no sea HTML
            primeros_bytes = f.read(100)
            f.seek(0)  # Volver al inicio
            
            # Si empieza con '<', probablemente es HTML (error de Google Drive)
            if primeros_bytes.startswith(b'<'):
                return False, "El archivo parece ser HTML en lugar de un modelo PKL"
            
            # Intentar cargar el pickle para verificar que es válido
            pickle.load(f)
            return True, "Archivo PKL válido"
    except Exception as e:
        return False, f"Error validando PKL: {str(e)}"

# Función para descargar un archivo si no existe
def descargar_modelo(url, path):
    archivo_existe = os.path.exists(path)
    archivo_valido = False
    
    if archivo_existe:
        if path.endswith('.pkl'):
            archivo_valido, mensaje = verificar_archivo_pkl(path)
            if not archivo_valido:
                print(f"⚠️ {os.path.basename(path)} existe pero está corrupto: {mensaje}")
                print(f"🔄 Re-descargando {os.path.basename(path)}...")
                os.remove(path)  # Eliminar archivo corrupto
                archivo_existe = False
        else:
            archivo_valido = True  # Para CSVs e imágenes, asumir que están bien si existen
    
    if not archivo_existe or not archivo_valido:
        print(f"📥 Descargando {os.path.basename(path)}...")
        try:
            # Utilizamos gdown para descargar el archivo
            # gdown acepta tanto URLs tipo /file/d/ID como /uc?id=ID
            gdown.download(url, path, quiet=False)
            
            # Verificar que el archivo descargado sea válido (solo para PKL)
            if path.endswith('.pkl'):
                es_valido, mensaje = verificar_archivo_pkl(path)
                if es_valido:
                    print(f"✅ {os.path.basename(path)} descargado y validado correctamente.")
                else:
                    print(f"❌ {os.path.basename(path)} descargado pero no es válido: {mensaje}")
                    print("💡 Verifica que la URL de Google Drive sea pública y correcta.")
            else:
                print(f"✅ {os.path.basename(path)} descargado correctamente.")
                
        except Exception as e:
            print(f"❌ Error en la descarga de {os.path.basename(path)}: {e}")
            print("💡 Verifica que la URL de Google Drive sea pública y accesible.")
    else:
        print(f"✅ {os.path.basename(path)} ya existe y es válido.")

# Descargar todos los archivos listados en files_to_download (si no existen)
st.info("🔄 Verificando y descargando archivos necesarios...")
archivos_descargados = []
archivos_con_error = []

for url, filename in files_to_download.items():
    target_path = os.path.join(modelos_dir, filename)
    try:
        descargar_modelo(url, target_path)
        if os.path.exists(target_path):
            archivos_descargados.append(filename)
        else:
            archivos_con_error.append(filename)
    except Exception as e:
        archivos_con_error.append(f"{filename} (Error: {str(e)})")

# Mostrar resumen de descargas
if archivos_descargados:
    st.success(f"✅ Archivos disponibles: {len(archivos_descargados)}")
if archivos_con_error:
    st.warning(f"⚠️ Archivos con problemas: {len(archivos_con_error)}")
    with st.expander("Ver detalles de archivos problemáticos"):
        for archivo in archivos_con_error:
            st.write(f"- {archivo}")
        
        st.markdown("### 🔧 Soluciones recomendadas:")
        st.markdown("""
        1. **Verificar permisos en Google Drive:**
           - Los archivos deben ser públicos (compartir con "cualquier persona con el enlace")
           - El enlace debe ser directo al archivo, no a una carpeta
        
        2. **Formato correcto de URL:**
           - Usar: `https://drive.google.com/file/d/ID_DEL_ARCHIVO`
           - No usar: `https://drive.google.com/open?id=...` o enlaces de vista previa
        
        3. **Si persiste el problema:**
           - Re-deployar la aplicación en Streamlit para limpiar caché
           - Verificar que los archivos PKL no estén corruptos en Google Drive
        """)

# Función para cargar los modelos en caché usando st.cache_resource
@st.cache_resource
def cargar_modelo(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

# Función para cargar modelos locales (DESDE CARPETA MODELOS)
@st.cache_resource
def cargar_modelo_local(nombre_modelo):
    """Carga modelos desde la carpeta modelos con validación robusta"""
    try:
        # Ruta hacia la carpeta modelos
        ruta = os.path.join(modelos_dir, f"{nombre_modelo}.pkl")
        
        if not os.path.exists(ruta):
            st.error(f"❌ No se encontró el modelo: {ruta}")
            return None
        
        # Verificar tamaño del archivo
        tamaño_archivo = os.path.getsize(ruta)
        if tamaño_archivo < 1000:  # Menos de 1KB probablemente está corrupto
            st.error(f"❌ {nombre_modelo}: Archivo muy pequeño ({tamaño_archivo} bytes), probablemente corrupto")
            return None
        
        # Verificar que no sea HTML (error de Google Drive)
        with open(ruta, 'rb') as f:
            primeros_bytes = f.read(100)
            if primeros_bytes.startswith(b'<'):
                st.error(f"❌ {nombre_modelo}: El archivo descargado es HTML en lugar de un modelo PKL")
                st.info("💡 La URL de Google Drive podría no ser pública o estar mal configurada")
                return None
        
        # Cargar el modelo
        with open(ruta, 'rb') as f:
            modelo = pickle.load(f)
            st.success(f"✅ {nombre_modelo} cargado correctamente")
            return modelo
            
    except pickle.UnpicklingError as e:
        st.error(f"❌ {nombre_modelo}: Error de deserialización - {str(e)}")
        st.info("💡 El archivo PKL podría estar corrupto o ser incompatible")
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
    
    # Información adicional sobre archivos descargados
    st.markdown("### 📁 Estado de archivos en carpeta modelos:")
    try:
        archivos_modelos = os.listdir(modelos_dir)
        for archivo in sorted(archivos_modelos):
            if archivo.endswith('.pkl'):
                ruta_completa = os.path.join(modelos_dir, archivo)
                tamaño = os.path.getsize(ruta_completa) / 1024  # KB
                
                # Verificar si es un archivo válido
                if tamaño < 1:
                    estado = "❌ Muy pequeño (corrupto)"
                else:
                    try:
                        with open(ruta_completa, 'rb') as f:
                            primeros_bytes = f.read(10)
                            if primeros_bytes.startswith(b'<'):
                                estado = "❌ HTML (error de Drive)"
                            else:
                                estado = "✅ Formato correcto"
                    except:
                        estado = "❌ Error al leer"
                
                st.write(f"- **{archivo}** ({tamaño:.2f} KB) - {estado}")
    except Exception as e:
        st.write(f"Error listando archivos: {e}")

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

# Función para cargar CSVs con manejo robusto de errores
@st.cache_data
def cargar_csv_seguro(ruta_archivo, nombre_archivo):
    """Carga un CSV con manejo robusto de errores"""
    try:
        # Verificar que el archivo existe
        if not os.path.exists(ruta_archivo):
            st.error(f"❌ Archivo no encontrado: {nombre_archivo}")
            return pd.DataFrame()
        
        # Mostrar información del archivo
        file_size = os.path.getsize(ruta_archivo) / (1024 * 1024)  # MB
        st.info(f"📄 Cargando {nombre_archivo} ({file_size:.2f} MB)")
        
        # Intentar carga normal primero
        df = pd.read_csv(ruta_archivo)
        st.success(f"✅ {nombre_archivo} cargado correctamente ({df.shape[0]} filas, {df.shape[1]} columnas)")
        return df
        
    except pd.errors.ParserError as e:
        st.warning(f"⚠️ Error de parsing en {nombre_archivo}: {str(e)}")
        st.info("🔧 Intentando métodos alternativos de carga...")
        
        # Método 1: Intentar con sep=None y engine='python'
        try:
            df = pd.read_csv(ruta_archivo, sep=None, engine='python', on_bad_lines='skip')
            st.info(f"✅ {nombre_archivo} cargado con separador automático ({df.shape[0]} filas, {df.shape[1]} columnas)")
            return df
        except Exception:
            pass
        
        # Método 2: Intentar con delimitador de coma y manejo de errores
        try:
            df = pd.read_csv(ruta_archivo, sep=',', on_bad_lines='skip', quoting=1)
            st.info(f"✅ {nombre_archivo} cargado omitiendo líneas problemáticas ({df.shape[0]} filas, {df.shape[1]} columnas)")
            return df
        except Exception:
            pass
        
        # Método 3: Intentar leyendo solo las primeras líneas para diagnóstico
        try:
            df_sample = pd.read_csv(ruta_archivo, nrows=10, on_bad_lines='skip')
            st.warning(f"⚠️ Solo se pudieron cargar las primeras 10 filas de {nombre_archivo}")
            st.info(f"📊 Muestra: {df_sample.shape[0]} filas, {df_sample.shape[1]} columnas")
            return df_sample
        except Exception:
            pass
        
        st.error(f"❌ No se pudo cargar {nombre_archivo} con ningún método")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"❌ Error inesperado cargando {nombre_archivo}: {str(e)}")
        return pd.DataFrame()

# Cargar los archivos CSV con manejo robusto de errores
df = cargar_csv_seguro("modelos/archivo.csv", "archivo.csv")
df2 = cargar_csv_seguro("modelos/dataset_LIMPIO_original.csv", "dataset_LIMPIO_original.csv")

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

    # Verificar que los DataFrames no estén vacíos antes de proceder
    if df.empty or df2.empty:
        st.error("❌ No se pudieron cargar los archivos CSV necesarios para las visualizaciones.")
        st.info("📊 Archivos requeridos: archivo.csv y dataset_LIMPIO_original.csv")
        
        if df.empty:
            st.warning("⚠️ archivo.csv no disponible")
        if df2.empty:
            st.warning("⚠️ dataset_LIMPIO_original.csv no disponible")
        
        # Mostrar información de debugging
        with st.expander("🔍 Información de Debugging"):
            st.write("**Archivos en la carpeta modelos:**")
            try:
                archivos_modelos = os.listdir("modelos")
                for archivo in archivos_modelos:
                    ruta_completa = os.path.join("modelos", archivo)
                    if os.path.isfile(ruta_completa):
                        tamaño = os.path.getsize(ruta_completa) / 1024  # KB
                        st.write(f"- {archivo} ({tamaño:.2f} KB)")
            except Exception as e:
                st.write(f"Error listando archivos: {e}")
                
        st.info("💡 **Solución temporal:** La aplicación continuará funcionando con las otras pestañas (modelos de ML).")
    
    else:
        # Solo ejecutar visualizaciones si ambos DataFrames están disponibles
        try:
            C_visualizacion(df, df2)
        except Exception as e:
            st.error(f"❌ Error en visualizaciones: {str(e)}")
            st.info("📊 Las visualizaciones no están disponibles temporalmente.")

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
