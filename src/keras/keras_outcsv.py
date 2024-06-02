import csv
import tensorflow as tf
import numpy as np
import keras
import sys

from keras.models import load_model, model_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


import csv


#============================--Fonctions--============================

#Permet d'ecrire dans un tableau CSV
def ecrire_tableau_dans_csv(nom_fichier, tableau):
    with open(nom_fichier, 'w', newline='') as fichier_csv:
        writer = csv.writer(fichier_csv)
        writer.writerows(tableau)




target_size = (47, 63)

#Charge l'image et la traite pour pouvoir la predire
def load_and_preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normaliser l'image
    return img_array

#Fonction quiretourne la prediction de l'image
def predict_image(model, image_path, target_size):
    img_array = load_and_preprocess_image(image_path, target_size)
    num_pixels = img_array.shape[1] * img_array.shape[2]*img_array.shape[3]
    img_array = img_array.reshape(img_array.shape[0], num_pixels)
    prediction = model.predict(img_array)
    return prediction

#Permet de lire les fichiers par lignes
def read_lines_from_file(file_path):
    lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        # Enlever les caractères de fin de ligne (\n)
        lines = [line.strip() for line in lines]
    except FileNotFoundError:
        print(f"Le fichier {file_path} n'a pas été trouvé.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
    return lines

#Permet de retourner la liste des labels de prediction de la plus probable à la moins probable
def sortPredict(prediction):
    pred = prediction[0]
    pred = np.argsort(pred)
    pred = np.flip(pred)
    return pred


#============================--MAIN--============================

#-----===Chargement des models et gestion des arguments===-----

if len(sys.argv) < 3:
        print("Usage: python keras_outcsv.py <fichier liste des chemins> <fichier des labels> <nom fichier csv> <(optionel) nom de model>")
        sys.exit(1)
if len(sys.argv) < 4 : 
    model = load_model(sys.argv[4])
else : 
    #Charge le model
    model = load_model("model.h5")

    #Charge l'architecture et les poids séparément
    """
    with open('model-traited.json', 'r') as json_file:
        model_json = json_file.read()
        model = model_from_json(model_json)
    model.load_weights('model-traited-poids.h5')
    """

file_path = sys.argv[1]
label_path = sys.argv[2]
nomCSV = sys.argv[3]
lines = read_lines_from_file(file_path)
lineslabels = read_lines_from_file(label_path)



N= 10
donnees = [["Expected value"]]
for k in range(N) : 
    donnees[0].append(k)

#-----===Lecture des chemins des images et predictions===-----
i=0
n=0
for image_path in lines :  
    label = lineslabels.__getitem__(i)
    prediction = predict_image(model, image_path, target_size)
    prediction = sortPredict(prediction)
    prediction = prediction[:N]
    i+=1

    if(int(label) == prediction[0]) :
        n+=1
    print("Prédiction num ", i ," : \n", prediction, "\ ", int(label), "-> ", prediction[0] ,"\n")
    donnees.append(np.insert(prediction, 0, label))

print("Prediction accuracy = ", n , "/", i)
ecrire_tableau_dans_csv(nomCSV, donnees)