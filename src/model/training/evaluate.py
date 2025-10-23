    #//src/model/training/evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import os
from tqdm import tqdm
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Usar un backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

from dataset import CBISDDSMDataset

def evaluate_model():
    """
    Función principal para evaluar el modelo entrenado en el conjunto de prueba.
    """
    # --- 1. Configuración ---
    PROJECT_ROOT = 'C:/Users/Alexi/Desktop/proyecto-cancer-mama'
    CLEAN_CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'metadata_clean.csv')
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model.pth')
    REPORTS_PATH = os.path.join(PROJECT_ROOT, 'reports')
    os.makedirs(REPORTS_PATH, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    NUM_WORKERS = 0

    print(f"--- Iniciando Evaluación en el Dispositivo: {DEVICE} ---")

    # --- 2. Cargar Datos de Prueba ---
    # Transformaciones para test (deben coincidir con entrenamiento)
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Cargando dataset de prueba...")
    test_dataset = CBISDDSMDataset(csv_file=CLEAN_CSV_PATH, split='test', transform=test_transforms)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- 3. Cargar el Modelo Entrenado ---
    print(f"Cargando modelo desde: {MODEL_PATH}")
    model = models.efficientnet_b4(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    model = model.to(DEVICE)
    model.eval()

    # --- 4. Definir mejor umbral encontrado en validación ---
    # Este valor debe coincidir con el que encontró el entrenamiento (ejemplo: 0.20)
    best_threshold = 0.20  # Usa el mejor umbral calculado en entrenamiento/validación

    # --- 5. Realizar Predicciones
    all_labels = []
    all_preds = []
    all_outputs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluando en el conjunto de prueba"):
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            outputs = model(images)
            all_outputs.extend(outputs.cpu().numpy().flatten())
            predicted = (outputs > best_threshold).float()
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # --- 6. Calcular y Mostrar Métricas ---
    print("\n--- Resultados de la Evaluación ---")
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_outputs)

    cm = confusion_matrix(all_labels, all_preds)

    # Cálculo de la especificidad (True Negative Rate)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else: # Caso borde si el modelo solo predice una clase
        specificity = 0.0
        print("Advertencia: La matriz de confusión no es 2x2. La especificidad podría no ser precisa.")

    # Mostrar métricas
    print(f"Precisión (Accuracy): {accuracy:.4f}")
    print(f"Precisión (Precision): {precision:.4f}")
    print(f"Sensibilidad (Recall): {recall:.4f}")
    print(f"Puntuación F1: {f1:.4f}")
    print(f"Especificidad: {specificity:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

    # --- 7. Generar y Guardar Matriz de Confusión ---
    print("\nGenerando matriz de confusión...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benigno', 'Maligno'], 
                yticklabels=['Benigno', 'Maligno'])
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.title('Matriz de Confusión')

    confusion_matrix_path = os.path.join(REPORTS_PATH, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)

    print(f"✅ Matriz de confusión guardada en: {confusion_matrix_path}")


if __name__ == '__main__':
    evaluate_model()


