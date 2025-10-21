# //src/backend/routes/model.py
# Este archivo contiene el Blueprint para todas las rutas relacionadas con el modelo de IA.

from flask import Blueprint, request, jsonify, url_for, current_app
import os
import uuid
# --- MODIFICACIÓN: Usar la importación directa ---
from inference import ModelInference
# --- FIN DE LA MODIFICACIÓN ---

# --- 1. Creación del Blueprint ---
model_bp = Blueprint('model', __name__)

# --- 2. Cargar el Modelo UNA SOLA VEZ ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model.pth')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"El archivo del modelo no se encuentra en: {MODEL_PATH}")

inference_engine = ModelInference(model_path=MODEL_PATH)

# --- 3. Definir el Endpoint /predict ---
@model_bp.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para recibir una imagen DICOM y obtener una predicción con explicación.
    """
    # --- INICIO DEL BLOQUE DE DEPURACIÓN ---
    # Imprimimos en la terminal del servidor para ver qué está llegando.
    print("\n--- INICIO DEPURACIÓN DE LA PETICIÓN ---")
    print(f"Ruta de la petición: {request.path}")
    print("Cabeceras (Headers):")
    print(request.headers)
    print("Archivos recibidos (request.files):")
    print(request.files)
    print("--- FIN DEPURACIÓN ---\n")
    # --- FIN DEL BLOQUE DE DEPURACIÓN ---

    if 'file' not in request.files:
        return jsonify({"error": "No se encontró ningún archivo en la petición."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "El archivo seleccionado no tiene nombre."}), 400

    if file:
        image_bytes = file.read()
        
        explanations_folder = os.path.join(current_app.static_folder, 'explanations')
        os.makedirs(explanations_folder, exist_ok=True)
        
        unique_filename = f"{uuid.uuid4()}.png"
        save_path = os.path.join(explanations_folder, unique_filename)
        
        result = inference_engine.predict_and_explain(dicom_file_bytes=image_bytes, save_path=save_path)
        
        if "error" in result:
            return jsonify(result), 500
        
        image_url = url_for('static', filename=f'explanations/{unique_filename}', _external=True)
        result['explanation_url'] = image_url
        
        return jsonify(result)

