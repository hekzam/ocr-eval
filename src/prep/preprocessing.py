import os
import cv2
import numpy as np

# Fonction pour traiter une image
def traiter_image(image_path, output_folder):
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Supprimer les bordures (utilisez vos propres paramètres)
    cropped_image = image[5:-5, 5:-5]

    # Binariser l'image
    _, binarized_image = cv2.threshold(cropped_image, 200, 255, cv2.THRESH_BINARY)

    # Réduire le bruit (utilisez vos propres paramètres)
    # denoised_image = cv2.fastNlMeansDenoising(binarized_image, None, 10, 7, 15)
    denoised_image = binarized_image

    # Redimensionner à 200-300 dpi
    dpi = 200  # Changer si nécessaire
    scale_percent = dpi / (image.shape[0] / 25.4)  # Convertir de pixels/mm à pixels/pouce
    width = int(denoised_image.shape[1] * scale_percent / 25.4)
    height = int(denoised_image.shape[0] * scale_percent / 25.4)
    resized_image = cv2.resize(denoised_image, (width, height), interpolation=cv2.INTER_AREA)

    # Enregistrer l'image traitée
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, resized_image)

# Dossier d'entrée et de sortie
input_folder = "parsed-200dpi/OCR_digits"
output_folder = "parsed-200dpi-treated/OCR_digits"

# Créer le dossier de sortie s'il n'existe pas
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Traiter chaque image dans le dossier d'entrée
for i in range(5):
    current_folder = input_folder + "/page_" + str(i)
    current_output_folder = output_folder + "/page_" + str(i)
    if not os.path.exists(current_output_folder):
        os.makedirs(current_output_folder)

    for filename in os.listdir(current_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(current_folder, filename)
            traiter_image(image_path, current_output_folder)
