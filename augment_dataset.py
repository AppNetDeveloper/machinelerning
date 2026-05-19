"""
Script de augmentación para expandir el dataset.
Genera variaciones de las imágenes existentes hasta alcanzar el objetivo.

Uso:
    python augment_dataset.py              # Default: 400 fotos por clase
    python augment_dataset.py --target 300 # 300 fotos por clase
    python augment_dataset.py --target 500 # 500 fotos por clase
"""

import argparse
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np

from config import DATASET_DIR

TARGET_DEFAULT = 400
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def get_existing_images(class_dir: Path) -> list[Path]:
    """Obtiene lista de imágenes existentes en la carpeta."""
    return [f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in EXTENSIONS]


def augment_image(img: Image.Image) -> Image.Image:
    """Aplica una combinación aleatoria de augmentaciones a una imagen."""
    augmentations = [
        lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
        lambda im: im.rotate(random.uniform(-25, 25), expand=False, fillcolor=(0, 0, 0)),
        lambda im: ImageEnhance.Brightness(im).enhance(random.uniform(0.7, 1.3)),
        lambda im: ImageEnhance.Contrast(im).enhance(random.uniform(0.7, 1.3)),
        lambda im: ImageEnhance.Color(im).enhance(random.uniform(0.7, 1.3)),
        lambda im: ImageEnhance.Sharpness(im).enhance(random.uniform(0.5, 1.5)),
        lambda im: im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5))),
        lambda im: ImageOps.posterize(im, bits=random.choice([4, 5, 6])),
        lambda im: ImageOps.solarize(im, threshold=random.randint(128, 200)),
        lambda im: im.transform(im.size, Image.AFFINE,
                                (1, random.uniform(-0.1, 0.1), random.randint(-10, 10),
                                 random.uniform(-0.1, 0.1), 1, random.randint(-10, 10)),
                                fillcolor=(0, 0, 0)),
        lambda im: ImageEnhance.Brightness(im).enhance(random.uniform(0.9, 1.1)).rotate(
            random.uniform(-10, 10), fillcolor=(0, 0, 0)),
    ]

    # Aplicar 2-4 augmentaciones aleatorias
    num_aug = random.randint(2, 4)
    selected = random.sample(augmentations, min(num_aug, len(augmentations)))

    result = img.copy()
    for aug in selected:
        try:
            result = aug(result)
        except Exception:
            continue

    return result


def augment_class(class_dir: Path, target: int):
    """Augmenta imágenes de una clase hasta alcanzar el target."""
    existing = get_existing_images(class_dir)
    current = len(existing)

    if current >= target:
        print(f"  {class_dir.name}: {current} imagenes (ya cumple objetivo)")
        return

    needed = target - current
    print(f"  {class_dir.name}: {current} imagenes -> generando {needed} mas...")

    generated = 0
    attempts = 0
    max_attempts = needed * 3  # Evitar loop infinito

    while generated < needed and attempts < max_attempts:
        attempts += 1

        # Seleccionar imagen original aleatoria
        src = random.choice(existing)
        try:
            img = Image.open(src).convert('RGB')
        except Exception:
            continue

        # Aplicar augmentación
        aug_img = augment_image(img)

        # Guardar con nombre único
        aug_name = f"aug_{generated:04d}_{src.stem}.jpg"
        aug_path = class_dir / aug_name

        # Evitar sobrescribir
        if aug_path.exists():
            continue

        aug_img.save(str(aug_path), 'JPEG', quality=90)
        generated += 1

    final = len(get_existing_images(class_dir))
    print(f"  {class_dir.name}: {final} imagenes (generadas: {generated})")


def main():
    parser = argparse.ArgumentParser(description="Augmentar dataset de confecciones")
    parser.add_argument("--target", type=int, default=TARGET_DEFAULT,
                        help=f"Numero objetivo de imagenes por clase (default: {TARGET_DEFAULT})")
    parser.add_argument("--min", type=int, default=300,
                        help="Minimo de imagenes por clase (default: 300)")
    args = parser.parse_args()

    print(f"=" * 50)
    print(f"  Augmentacion del Dataset")
    print(f"  Objetivo: {args.target} imagenes por clase")
    print(f"  Minimo: {args.min} imagenes por clase")
    print(f"=" * 50)

    if not DATASET_DIR.exists():
        print("ERROR: No existe la carpeta dataset/")
        return

    classes = [d for d in sorted(DATASET_DIR.iterdir()) if d.is_dir()]

    if not classes:
        print("ERROR: No hay clases en el dataset")
        return

    print(f"\nClases encontradas: {len(classes)}")

    for class_dir in classes:
        augment_class(class_dir, args.target)

    # Resumen final
    print(f"\n{'=' * 50}")
    print("Resumen final:")
    total = 0
    for class_dir in classes:
        count = len(get_existing_images(class_dir))
        total += count
        print(f"  {class_dir.name}: {count} imagenes")
    print(f"  Total: {total} imagenes")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
