# //src/utils/extract_test_samples.py
# Este script sirve para extraer un pequeño conjunto de imágenes de prueba
# (benignas y malignas) del dataset completo y copiarlas a una nueva carpeta
# para facilitar las pruebas manuales o la depuración.

import pandas as pd
import os
import shutil
from tqdm import tqdm

def extract_samples(num_samples=20): # <-- Se cambió el valor por defecto
    """
    Copia un número específico de imágenes de prueba benignas y malignas
    a una nueva carpeta 'test_samples/'.
    """
    # --- 1. Configuración de Rutas ---
    # Asume que el script se ejecuta desde la raíz del proyecto.
    PROJECT_ROOT = 'C:/Users/Alexi/Desktop/proyecto-cancer-mama'
    CLEAN_CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'metadata_clean.csv')
    DEST_FOLDER = os.path.join(PROJECT_ROOT, 'test_samples')

    print(f"--- Creando carpeta de destino en: {DEST_FOLDER} ---")
    os.makedirs(DEST_FOLDER, exist_ok=True)

    # --- 2. Cargar y Filtrar Datos ---
    try:
        df = pd.read_csv(CLEAN_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{CLEAN_CSV_PATH}'.")
        print("Asegúrate de haber ejecutado primero el script 'create_clean_dataframe.py'.")
        return

    # Filtrar solo los datos del conjunto de prueba
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    # Seleccionar muestras
    benign_samples = test_df[test_df['pathology'] == 'BENIGN'].head(num_samples)
    malignant_samples = test_df[test_df['pathology'] != 'BENIGN'].head(num_samples)
    
    if len(benign_samples) < num_samples or len(malignant_samples) < num_samples:
        print(f"ADVERTENCIA: No se encontraron suficientes muestras. Se copiarán {len(benign_samples)} benignas y {len(malignant_samples)} malignas.")

    # --- 3. Copiar Archivos ---
    print(f"Copiando {len(benign_samples)} imágenes benignas...")
    for index, row in tqdm(benign_samples.iterrows(), total=len(benign_samples)):
        source_path = row['correct_image_path']
        dest_filename = f"benign_sample_{index+1}.dcm"
        dest_path = os.path.join(DEST_FOLDER, dest_filename)
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
        else:
            print(f"  - Advertencia: No se encontró el archivo de origen: {source_path}")

    print(f"Copiando {len(malignant_samples)} imágenes malignas...")
    for index, row in tqdm(malignant_samples.iterrows(), total=len(malignant_samples)):
        source_path = row['correct_image_path']
        dest_filename = f"malignant_sample_{index+1}.dcm"
        dest_path = os.path.join(DEST_FOLDER, dest_filename)
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
        else:
            print(f"  - Advertencia: No se encontró el archivo de origen: {source_path}")
            
    print(f"\n✅ ¡Éxito! {len(benign_samples) + len(malignant_samples)} imágenes de prueba han sido copiadas a la carpeta '{DEST_FOLDER}'.")

if __name__ == '__main__':
    # Puedes cambiar el número de muestras aquí si lo deseas
    extract_samples(num_samples=20) 

