import numpy as np
from PIL import Image
import os

img_size = 128
base_dir = 'flowers'
categories = ['dandelion', 'daisy', 'rose', 'sunflower', 'tulip']


def load_and_save_image(image_path, output_dir):
    # Wczytaj i przeskaluj obraz
    image = Image.open(image_path)
    image = image.resize((img_size, img_size))
    image = np.array(image, dtype=np.float32) / 255  # Normalizacja jako float32

    # Tworzenie ścieżki wyjściowej, jeśli nie istnieje
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ustawienie nazwy pliku wyjściowego
    output_path = os.path.join(output_dir, os.path.basename(image_path) + '.npy')

    # Zapisz znormalizowany obraz jako .npy, zachowując typ float32
    np.save(output_path, image)


for category in categories:
    category_path = os.path.join(base_dir, category)
    output_category_path = os.path.join(base_dir, category + '_normalized')

    for subdir, dirs, files in os.walk(category_path):
        for file in files:
            # Ignorowanie plików, które nie są obrazami .jpg
            if file.lower().endswith(".jpg"):
                file_path = os.path.join(subdir, file)
                load_and_save_image(file_path, output_category_path)