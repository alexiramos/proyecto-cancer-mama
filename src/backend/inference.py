# //src/backend/inference.py
# Este script es el "motor" de la IA para el backend.
# Carga el modelo entrenado y la lógica de Grad-CAM para procesar nuevas imágenes.

import torch
import torch.nn as nn
from torchvision import models, transforms
import pydicom
import numpy as np
import cv2
import os
from io import BytesIO

# --- Implementación de Grad-CAM ---
class GradCAM:
    """
    Clase para generar mapas de activación de gradiente (Grad-CAM).
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hook_layers()

    def _hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_image):
        self.model.eval()
        output = self.model(input_image)
        
        self.model.zero_grad()
        output.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()
        
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = torch.nn.functional.relu(heatmap)
        
        heatmap_max = torch.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max
            
        return heatmap.cpu().numpy(), output.item()


class ModelInference:
    """
    Encapsula toda la lógica para cargar el modelo y realizar predicciones.
    """
    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.transforms = self._get_transforms()
        self.grad_cam = GradCAM(self.model, self.model.features[-1])
        print("✅ Motor de inferencia con Grad-CAM listo.")

    def _load_model(self, model_path):
        """Carga el modelo EfficientNet-B4 entrenado."""
        print(f"Cargando modelo desde: {model_path}")
        model = models.efficientnet_b4(weights=None) 
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def _get_transforms(self):
        """Define las transformaciones, usando el tamaño de imagen óptimo (380x380)."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((380, 380)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _preprocess_dicom(self, dicom_file_bytes):
        """Lee los bytes de un archivo DICOM y lo convierte a una imagen RGB."""
        try:
            dicom_file = BytesIO(dicom_file_bytes)
            dicom = pydicom.dcmread(dicom_file)
            image = dicom.pixel_array
            
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            return image
        except Exception as e:
            print(f"Error al procesar el archivo DICOM: {e}")
            return None

    def predict_and_explain(self, dicom_file_bytes, save_path):
        """
        Realiza una predicción y genera la explicación visual.
        """
        original_image = self._preprocess_dicom(dicom_file_bytes)
        if original_image is None:
            return {"error": "No se pudo procesar el archivo DICOM."}

        image_tensor = self.transforms(original_image).unsqueeze(0).to(self.device)

        heatmap, confidence = self.grad_cam.generate_heatmap(image_tensor)
        
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        original_image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        superimposed_img = cv2.addWeighted(original_image_bgr, 0.6, heatmap_colored, 0.4, 0)
        
        cv2.imwrite(save_path, superimposed_img)
        
        # --- MODIFICACIÓN CLAVE ---
        # Se ajustan las claves y el formato del resultado final.
        prediction = 1 if confidence > 0.5 else 0
        diagnosis = "maligno" if prediction == 1 else "benigno"
        result = {
            "prediction": diagnosis,
            "probability": float(f"{confidence:.4f}")
        }
        # --- FIN DE LA MODIFICACIÓN ---
        return result

