    #//src/model/training/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, recall_score, roc_auc_score, confusion_matrix, precision_score

from dataset import CBISDDSMDataset

def collate_fn_skip_none(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

def find_best_threshold(y_true, y_probs):
    best_f1, best_th = 0, 0.5
    thresholds = np.linspace(0.2, 0.8, 21)
    for th in thresholds:
        y_pred = (y_probs >= th).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_th, best_f1

def main():
    PROJECT_ROOT = 'C:/Users/Alexi/Desktop/proyecto-cancer-mama'
    CLEAN_CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'metadata_clean.csv')
    MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models') 
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16 
    NUM_WORKERS = 0

    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 20

    print(f"--- Entrenamiento EfficientNet-B4 en: {DEVICE} ---")

    # Transformaciones mejoradas
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop((380, 380), scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    print("Cargando datasets...")
    train_dataset = CBISDDSMDataset(csv_file=CLEAN_CSV_PATH, split='train', transform=data_transforms['train'])
    val_dataset = CBISDDSMDataset(csv_file=CLEAN_CSV_PATH, split='val', transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn_skip_none)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn_skip_none)

    print("Calculando pesos de clase...")
    df = pd.read_csv(CLEAN_CSV_PATH)
    train_df = df[df['split'] == 'train']
    neg_count = (train_df['pathology'] == 'BENIGN').sum()
    pos_count = (train_df['pathology'] != 'BENIGN').sum()
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float).to(DEVICE)
    print(f"Peso para la clase Maligno: {pos_weight.item():.2f}")

    print("Definiendo el modelo EfficientNet-B4...")
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[-4:].parameters():
        param.requires_grad = True
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    model = model.to(DEVICE)

    criterion = nn.BCELoss(weight=pos_weight)
    params_to_update = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.AdamW(params_to_update, lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_f1 = 0.0
    best_auc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Época {epoch+1}/{NUM_EPOCHS} ---")

        # Entrenamiento
        model.train()
        for images, labels in tqdm(train_loader, desc="Entrenamiento"):
            if images.nelement() == 0:
                continue
            images, labels = images.to(DEVICE), labels.to(DEVICE).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validación extendida
        model.eval()
        all_val_labels = []
        all_val_outputs = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validación"):
                if images.nelement() == 0:
                    continue
                images, labels = images.to(DEVICE), labels.to(DEVICE).float()
                outputs = model(images)
                all_val_outputs.extend(outputs.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        all_val_labels = np.array(all_val_labels).reshape(-1)
        all_val_outputs = np.array(all_val_outputs).reshape(-1)
        best_th, best_f1_this = find_best_threshold(all_val_labels, all_val_outputs)
        predicted = (all_val_outputs >= best_th).astype(float)

        val_acc = (predicted == all_val_labels).mean()
        val_recall = recall_score(all_val_labels, predicted, zero_division=0)
        val_f1 = f1_score(all_val_labels, predicted, zero_division=0)
        val_auc = roc_auc_score(all_val_labels, all_val_outputs)
        val_prec = precision_score(all_val_labels, predicted, zero_division=0)
        conf = confusion_matrix(all_val_labels, predicted)
        tn, fp, fn, tp = conf.ravel() if conf.size == 4 else (0,0,0,0)
        val_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"Precisión (Accuracy): {val_acc:.4f} | Precisión: {val_prec:.4f} | Recall: {val_recall:.4f}")
        print(f"F1: {val_f1:.4f} | Especificidad: {val_spec:.4f} | AUC-ROC: {val_auc:.4f} | Mejor umbral: {best_th:.2f}")

        scheduler.step()

        # Guardar el mejor modelo según F1/AUC
        if val_f1 > best_f1 or val_auc > best_auc:
            best_f1 = max(val_f1, best_f1)
            best_auc = max(val_auc, best_auc)
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'best_model.pth'))
            print(f"Modelo guardado (F1: {best_f1:.4f}, AUC: {best_auc:.4f})")

    print("\n--- Entrenamiento finalizado ---")

if __name__ == '__main__':
    main()
