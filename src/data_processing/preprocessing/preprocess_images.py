#//src/data_processing/preprocessing/preprocess_images.py
# Este script se encarga del preprocesamiento completo:
# 1. Carga los metadatos de los archivos CSV.
# 2. Encuentra las rutas correctas de las imágenes DICOM.
# 3. Carga, procesa (redimensiona y normaliza) y etiqueta cada imagen de forma paralela.
# 4. Guarda los conjuntos de datos de entrenamiento y prueba en archivos CSV.

import pandas as pd
import pydicom
import os
import cv2  # OpenCV para el procesamiento de imágenes
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor # ¡NUEVA IMPORTACIÓN PARA PARALELISMO!

# ----------------------------------------------------
# Celda 1: Configuración de Rutas
# ----------------------------------------------------
PROJECT_ROOT = 'C:/Users/Alexi/Desktop/proyecto-cancer-mama'
IMAGES_BASE_PATH = 'G:/dataset/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM/'
DATA_RAW_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Asegurarse de que la carpeta de destino exista
os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)

# Definir el tamaño al que se redimensionarán todas las imágenes
IMG_SIZE = 224

# ----------------------------------------------------
# Celda 2: Funciones Auxiliares
# ----------------------------------------------------

def find_dicom_in_folder(folder_path):
    """Encuentra el primer archivo .dcm en el nivel superior de una carpeta."""
    if not os.path.isdir(folder_path):
        return None
    for item in os.listdir(folder_path):
        if item.lower().endswith('.dcm'):
            return os.path.join(folder_path, item)
    return None

def process_dicom_image(path):
    """Carga una imagen DICOM, la convierte a 8-bit, la redimensiona y la normaliza."""
    try:
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array
        
        # Normalizar y convertir a 8-bit (formato estándar para imágenes)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = (img * 255).astype(np.uint8)
        
        # Redimensionar la imagen al tamaño definido (e.g., 224x224)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        return img_resized
    except Exception as e:
        # Silenciamos el error para no inundar la consola durante el proceso paralelo
        # print(f"Error procesando la imagen {path}: {e}")
        return None

# --- FUNCIÓN MODIFICADA PARA USAR PARALELISMO ---
def load_images_and_labels(df, base_image_path):
    """Carga y procesa imágenes y etiquetas desde un DataFrame de forma paralela."""
    
    # Extraer las rutas y etiquetas del dataframe primero
    correct_paths_list = df['correct_image_path'].tolist()
    labels_list = [1 if 'MALIGNANT' in pathology.upper() else 0 for pathology in df['pathology'].tolist()]
    
    X, y, paths = [], [], []
    
    # Usar ThreadPoolExecutor para procesar imágenes en paralelo
    # Se usarán tantos hilos como núcleos de CPU tenga el sistema
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Usamos tqdm para visualizar el progreso del map
        # executor.map aplica la función 'process_dicom_image' a cada ruta en la lista
        results = list(tqdm(executor.map(process_dicom_image, correct_paths_list), total=len(correct_paths_list), desc="Procesando imágenes"))

    # Recolectar solo los resultados que se procesaron correctamente (no son None)
    for i, img in enumerate(results):
        if img is not None:
            X.append(img)
            y.append(labels_list[i])
            paths.append(correct_paths_list[i])
            
    return np.array(X), np.array(y), paths


def save_to_csv(X, y, paths, csv_filename):
    """
    Guarda los datos procesados en un archivo CSV.
    ADVERTENCIA: Guardar datos de imagen en CSV crea archivos muy grandes y es ineficiente.
                 Este método es solo para fines de demostración.
    """
    print(f"Guardando datos en '{csv_filename}'...")
    # Aplanar cada imagen para que sea una sola fila en el CSV
    X_flattened = X.reshape(X.shape[0], -1)
    
    df_data = pd.DataFrame(X_flattened)
    df_labels = pd.DataFrame({'label': y, 'path': paths})
    
    df_to_save = pd.concat([df_labels, df_data], axis=1)
    
    full_path = os.path.join(DATA_PROCESSED_PATH, csv_filename)
    df_to_save.to_csv(full_path, index=False)
    print(f"✅ Datos guardados con éxito en '{full_path}'")

# ----------------------------------------------------
# Celda 3: Lógica Principal de Ejecución
# ----------------------------------------------------

print("--- Paso 1: Cargando y combinando metadatos ---")

# Cargar datos de cáncer de masa
df_mass_train = pd.read_csv(os.path.join(DATA_RAW_PATH, 'mass_case_description_train_set.csv'))
df_mass_test = pd.read_csv(os.path.join(DATA_RAW_PATH, 'mass_case_description_test_set.csv'))

# Cargar datos de cáncer de calcificación
df_calc_train = pd.read_csv(os.path.join(DATA_RAW_PATH, 'calc_case_description_train_set.csv'))
df_calc_test = pd.read_csv(os.path.join(DATA_RAW_PATH, 'calc_case_description_test_set.csv'))

# Combinar los datos de entrenamiento y prueba
df_train = pd.concat([df_mass_train, df_calc_train], ignore_index=True)
df_test = pd.concat([df_mass_test, df_calc_test], ignore_index=True)

print("Metadatos cargados y combinados.")
print(f"Total de registros de entrenamiento: {len(df_train)}")
print(f"Total de registros de prueba: {len(df_test)}")


print("\n--- Paso 2: Buscando rutas de imágenes ---")
# Encontrar las rutas correctas para cada conjunto de datos
df_train['correct_image_path'] = df_train['image file path'].apply(
    lambda x: find_dicom_in_folder(os.path.join(IMAGES_BASE_PATH, x.strip().split('/')[0]))
)
df_test['correct_image_path'] = df_test['image file path'].apply(
    lambda x: find_dicom_in_folder(os.path.join(IMAGES_BASE_PATH, x.strip().split('/')[0]))
)
# Eliminar filas donde no se encontró la imagen
df_train.dropna(subset=['correct_image_path'], inplace=True)
df_test.dropna(subset=['correct_image_path'], inplace=True)

print("Rutas de imágenes verificadas.")

# 3. Cargar imágenes y etiquetas de entrenamiento y prueba y guardarlas en csv
print("\n--- Paso 3: Procesando el conjunto de ENTRENAMIENTO ---")
X_train, y_train, paths_train = load_images_and_labels(df_train, IMAGES_BASE_PATH)
csv_filenameTrain = 'resultadosXytrain.csv'
save_to_csv(X_train, y_train, paths_train, csv_filenameTrain)

print("\n--- Paso 4: Procesando el conjunto de PRUEBA ---")
X_test, y_test, paths_test = load_images_and_labels(df_test, IMAGES_BASE_PATH)
csv_filenameTest = 'resultadosXytest.csv'
save_to_csv(X_test, y_test, paths_test, csv_filenameTest)

print("\n--- Proceso de preprocesamiento completado. ---")





