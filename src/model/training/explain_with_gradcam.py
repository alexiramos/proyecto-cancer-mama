    #//src/model/training/explain_with_gradcam.py

import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Asumiendo el Dataset está en este archivo
from dataset import CBISDDSMDataset

# --- Implementación de Grad-CAM ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_in, grad_out):
            # <-- MEJORA 1: Usar el hook moderno y recomendado en PyTorch.
            self.gradients = grad_out[0]
            
        self.target_layer.register_forward_hook(forward_hook)
        # Se usa el hook más robusto.
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_image, class_idx=0): # class_idx por defecto a 0 para clasificación binaria
        self.model.eval()
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
            
        output = self.model(input_image)
        
        self.model.zero_grad()
        # La clase objetivo para la retropropagación.
        output[0][class_idx].backward(retain_graph=True)
        
        if self.gradients is None:
            raise RuntimeError("No se capturaron gradientes. Verifica la capa objetivo.")
            
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()
        
        # Ponderar los mapas de activación con los gradientes
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        # Aplicar ReLU para mantener solo las contribuciones positivas
        heatmap = torch.nn.functional.relu(heatmap)
        
        # Normalizar el mapa de calor para la visualización
        heatmap_max = torch.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max
            
        return heatmap.cpu().numpy()

def main():
    # --- 1. Configuración ---
    PROJECT_ROOT = 'C:/Users/Alexi/Desktop/proyecto-cancer-mama'
    CLEAN_CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'metadata_clean.csv')
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model.pth')
    REPORTS_PATH = os.path.join(PROJECT_ROOT, 'reports', 'grad_cam_examples_mejorado')
    os.makedirs(REPORTS_PATH, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {DEVICE}")

    # --- 2. Cargar Modelo (EfficientNet-B4) ---
    print("Cargando modelo EfficientNet-B4...")
    model = models.efficientnet_b4(weights=None)
    num_ftrs = model.classifier[1].in_features
    # Clasificador para una salida binaria (maligno/benigno)
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    model.to(DEVICE)
    model.eval()

    # --- 3. Definir Transformaciones ---
    # <-- CORRECCIÓN CRÍTICA: Usar el tamaño de imagen correcto para EfficientNet-B4.
    print("Definiendo transformaciones con el tamaño de imagen correcto (380x380)...")
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((380, 380)), # Este es el tamaño correcto
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Se carga el dataset sin transformaciones iniciales para obtener la imagen original
    test_dataset = CBISDDSMDataset(csv_file=CLEAN_CSV_PATH, split='test', transform=None)

    # --- 4. Seleccionar Muestras ---
    print("Seleccionando imágenes de ejemplo para Grad-CAM...")
    annotations_df = test_dataset.annotations
    
    benign_rows = annotations_df[annotations_df['pathology'].str.contains('BENIGN', case=False, na=False)]
    
    # <-- MEJORA 2: Lógica de selección más robusta para casos malignos.
    # Selecciona todo lo que no es benigno, cubriendo 'MALIGNANT' y otras posibles etiquetas.
    malignant_rows = annotations_df[~annotations_df['pathology'].str.contains('BENIGN', case=False, na=False)]
    
    # Tomar los índices de las primeras 2 filas de cada categoría
    benign_indices = benign_rows.index[:2].tolist()
    malignant_indices = malignant_rows.index[:2].tolist()
    sample_indices = benign_indices + malignant_indices
    
    if not sample_indices or len(sample_indices) < 4:
        print("ADVERTENCIA: No se encontraron suficientes muestras (2 benignas y 2 malignas).")
        return

    # --- 5. Generar y Guardar Mapas de Calor ---
    # La capa objetivo suele ser el último bloque convolucional del modelo
    grad_cam = GradCAM(model, model.features[-1])

    for idx in sample_indices:
        # Cargar la imagen original y su etiqueta
        original_image, label = test_dataset[idx]
        
        # Aplicar las transformaciones para alimentar el modelo
        input_tensor = test_transforms(original_image).to(DEVICE)
        
        # Generar el mapa de calor
        heatmap = grad_cam.generate_heatmap(input_tensor, class_idx=0)
        
        # Redimensionar el mapa de calor al tamaño de la imagen original
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # <-- MEJORA 3: Manejo seguro de imágenes en escala de grises.
        # Si la imagen original es grayscale, convertirla a BGR para la superposición.
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        # Superponer el mapa de calor sobre la imagen original
        # Un peso mayor a la imagen original (0.6) hace que sea más visible.
        superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

        # Guardar la imagen resultante
        label_text = "malignant" if label.item() == 1 else "benign"
        filename = f"sample_{idx}_{label_text}.png"
        save_path = os.path.join(REPORTS_PATH, filename)
        cv2.imwrite(save_path, superimposed_img)
        print(f"✅ Mapa de calor guardado en: {save_path}")

if __name__ == '__main__':
    main()

