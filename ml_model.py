"""
Gestor del modelo de ML.
Carga el modelo, maneja predicciones con TTA, y hot-reload cuando se reentrena.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from config import MODEL_PATH, CLASES_PATH, IMG_SIZE, TTA_AUGMENTATIONS


class ModelManager:
    """Gestor singleton del modelo de prediccion."""

    def __init__(self):
        self.model = None
        self.class_names = []
        self.model_mtime = 0.0
        self._setup_transforms()

    def _setup_transforms(self):
        """Configura las transformaciones para TTA."""
        self.base_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.tta_transforms = [
            self.base_transform,
            transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ColorJitter(brightness=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ColorJitter(brightness=0.8),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ColorJitter(contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        ]

    def load(self) -> bool:
        """Carga el modelo desde disco. Retorna True si se cargo correctamente."""
        try:
            if not MODEL_PATH.exists() or not CLASES_PATH.exists():
                return False

            with open(CLASES_PATH, "r") as f:
                self.class_names = json.load(f)

            num_classes = len(self.class_names)
            self.model = models.resnet50(weights=None)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            )

            self.model.load_state_dict(
                torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True)
            )
            self.model.eval()
            self.model_mtime = MODEL_PATH.stat().st_mtime
            return True

        except Exception as e:
            print(f"Error cargando modelo: {e}")
            self.model = None
            return False

    def check_reload(self) -> bool:
        """Verifica si el modelo cambio en disco y lo recarga si es necesario."""
        if not MODEL_PATH.exists():
            return False
        current_mtime = MODEL_PATH.stat().st_mtime
        if current_mtime > self.model_mtime:
            print("Detectado modelo actualizado, recargando...")
            return self.load()
        return False

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, image: Image.Image, use_tta: bool = True) -> dict:
        """Predice la clase de una imagen."""
        if not self.is_loaded:
            raise RuntimeError("Modelo no cargado")

        self.check_reload()

        if use_tta:
            probabilities = self._predict_tta(image)
        else:
            probabilities = self._predict_single(image, self.base_transform)

        confianza, predicted_idx = probabilities.max(0)
        confeccion = self.class_names[predicted_idx.item()]

        todas_probs = {
            self.class_names[i]: round(probabilities[i].item() * 100, 2)
            for i in range(len(self.class_names))
        }
        todas_probs = dict(sorted(todas_probs.items(), key=lambda x: x[1], reverse=True))

        return {
            "confeccion": confeccion,
            "confianza": round(confianza.item() * 100, 2),
            "probabilidades": todas_probs,
            "confiable": confianza.item() >= 0.5,
            "tta": use_tta,
        }

    def _predict_single(self, image: Image.Image, transform) -> torch.Tensor:
        """Prediccion con una sola transformacion."""
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
            return F.softmax(output, dim=1)[0]

    def _predict_tta(self, image: Image.Image) -> torch.Tensor:
        """Prediccion con Test-Time Augmentation."""
        all_probs = []
        for i in range(TTA_AUGMENTATIONS):
            t = self.tta_transforms[i % len(self.tta_transforms)]
            all_probs.append(self._predict_single(image, t))
        return torch.stack(all_probs).mean(dim=0)


# Instancia global del gestor de modelo
model_manager = ModelManager()
