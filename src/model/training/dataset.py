    #//src/model/training/dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import pydicom
import cv2
import numpy as np
from torchvision import transforms
import os
from sklearn.model_selection import train_test_split

class CBISDDSMDataset(Dataset):
    """
    Dataset personalizado para cargar las imágenes de mamografías CBIS-DDSM.
    Lee el archivo CSV limpio, carga las imágenes DICOM desde la ruta, y aplica transformaciones.
    Soporta la división en train, validation y test.
    """
    def __init__(self, csv_file, split='train', transform=None, val_size=0.15, random_state=42):
        """
        Args:
            csv_file (string): Ruta al archivo metadata_clean.csv.
            split (string): 'train', 'val', o 'test'.
            transform (callable, optional): Transformaciones a aplicar.
            val_size (float): Porcentaje del conjunto de entrenamiento original a usar para validación.
            random_state (int): Semilla para la reproducibilidad de la división.
        """
        all_data = pd.read_csv(csv_file)
        
        original_train_df = all_data[all_data['split'] == 'train']
        test_df = all_data[all_data['split'] == 'test']
        
        train_df, val_df = train_test_split(
            original_train_df,
            test_size=val_size,
            random_state=random_state,
            stratify=original_train_df['pathology']
        )

        if split == 'train':
            self.annotations = train_df.reset_index(drop=True)
        elif split == 'val':
            self.annotations = val_df.reset_index(drop=True)
        elif split == 'test':
            self.annotations = test_df.reset_index(drop=True)
        else:
            raise ValueError("El valor de 'split' debe ser 'train', 'val' o 'test'")
            
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = self.annotations.loc[index, 'correct_image_path']
        pathology = self.annotations.loc[index, 'pathology']
        
        label = 1 if 'MALIGNANT' in pathology.upper() else 0
        label = torch.tensor(label, dtype=torch.float32)

        dicom = pydicom.dcmread(img_path)
        image = dicom.pixel_array
        
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = (image * 255).astype(np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label.unsqueeze(-1)

