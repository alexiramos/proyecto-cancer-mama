    #//src/data_processing/preprocessing/create_clean_dataframe.py

import pandas as pd
import os
from tqdm import tqdm

# --- 1. Configuración de Rutas ---
PROJECT_ROOT = 'C:/Users/Alexi/Desktop/proyecto-cancer-mama'
IMAGES_BASE_PATH = 'G:/dataset/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM/'
DATA_RAW_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')

os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)

# --- 2. Función de Búsqueda Robusta ---

def find_dicom_recursively(folder_path):
    """
    Busca recursivamente el primer archivo .dcm en una carpeta y todas sus subcarpetas.
    Esta es la forma más robusta de encontrar los archivos.
    """
    if not os.path.isdir(folder_path):
        return None
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.dcm'):
                # Devolvemos la ruta completa y correcta.
                return os.path.join(root, file)
    return None # No se encontró ningún archivo .dcm

# --- 3. Lógica Principal ---

def create_clean_metadata():
    """
    Carga todos los metadatos, encuentra las rutas de imagen correctas de forma
    recursiva y guarda un único archivo CSV limpio.
    """
    print("--- Iniciando el preprocesamiento de metadatos (Versión Definitiva) ---")
    
    # Cargar los 4 archivos CSV
    df_mass_train = pd.read_csv(os.path.join(DATA_RAW_PATH, 'mass_case_description_train_set.csv'))
    df_mass_test = pd.read_csv(os.path.join(DATA_RAW_PATH, 'mass_case_description_test_set.csv'))
    df_calc_train = pd.read_csv(os.path.join(DATA_RAW_PATH, 'calc_case_description_train_set.csv'))
    df_calc_test = pd.read_csv(os.path.join(DATA_RAW_PATH, 'calc_case_description_test_set.csv'))

    # Añadir la columna 'split' para saber a qué conjunto pertenecen
    df_mass_train['split'] = 'train'
    df_mass_test['split'] = 'test'
    df_calc_train['split'] = 'train'
    df_calc_test['split'] = 'test'
    
    # Unificar todo en un solo DataFrame
    df = pd.concat([df_mass_train, df_mass_test, df_calc_train, df_calc_test], ignore_index=True)
    print(f"Se cargaron un total de {len(df)} registros.")

    print("Buscando las rutas de archivo DICOM correctas (esto puede tardar un poco)...")
    
    # Usar tqdm para mostrar una barra de progreso
    tqdm.pandas(desc="Verificando rutas")
    
    # Crear la ruta a la carpeta de cada paciente
    df['patient_folder'] = df['image file path'].apply(lambda x: os.path.join(IMAGES_BASE_PATH, x.strip().split('/')[0]))
    
    # Aplicar la búsqueda recursiva para encontrar la ruta correcta
    df['correct_image_path'] = df['patient_folder'].progress_apply(find_dicom_recursively)

    # Informar sobre archivos no encontrados y eliminarlos
    missing_files = df['correct_image_path'].isnull().sum()
    if missing_files > 0:
        print(f"\nADVERTENCIA: No se encontraron archivos DICOM para {missing_files} registros. Serán eliminados.")
        df.dropna(subset=['correct_image_path'], inplace=True)

    print(f"Se encontraron {len(df)} imágenes válidas.")

    # Seleccionar y renombrar columnas para el archivo final
    final_df = df[['patient_id', 'pathology', 'abnormality type', 'split', 'correct_image_path']].copy()
    
    # Guardar el archivo limpio
    output_path = os.path.join(DATA_PROCESSED_PATH, 'metadata_clean.csv')
    final_df.to_csv(output_path, index=False)
    
    print(f"\n✅ ¡Éxito! DataFrame limpio guardado en: '{output_path}'")
    print("\nPrimeras 5 filas del nuevo archivo:")
    print(final_df.head())

if __name__ == '__main__':
    create_clean_metadata()



