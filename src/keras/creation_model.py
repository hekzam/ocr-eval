from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

#==============----Fonction----==============

#Fonction de transformation
def dataset_to_numpy(dataset):
    images = []
    labels = []
    for img_batch, lbl_batch in dataset:
        images.append(img_batch.numpy())
        labels.append(lbl_batch.numpy())
    return np.concatenate(images), np.concatenate(labels)

#Fonction de creation des neuronnes
def neural_network():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    #model.add(Dropout(.5))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#==============----MAIN----==============

#gestion des arguments
if len(sys.argv) < 2:
        print("Usage: python keras_outcsv.py <chemins du repertoire d'entrainement>  <nom du model > < taille x default = 47>  <taille y default = 63>")
        sys.exit(1)
taillex=47
tailley=6
if len(sys.argv) > 4 : 
         taillex=sys.argv[3]
         tailley=sys.argv[4]

main_directory = sys.argv[1]
#creation du data set
full_dataset =  keras.utils.image_dataset_from_directory(
    main_directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=32,
    image_size=(taillex , tailley),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

# Calculer le nombre total de lots
total_batches = tf.data.experimental.cardinality(full_dataset).numpy()
test_batches = int(0.1 * total_batches)

# Diviser le dataset en ensembles de test et d'entraînement
test_dataset = full_dataset.take(test_batches)
train_dataset = full_dataset.skip(test_batches)

# Optionnel : Diviser l'ensemble d'entraînement en ensemble d'entraînement et de validation
validation_batches = int(0.1 * tf.data.experimental.cardinality(train_dataset).numpy())
validation_dataset = train_dataset.take(validation_batches)
train_dataset = train_dataset.skip(validation_batches)

# Fonction pour extraire les images et les labels d'un dataset

# Convertir les datasets en numpy arrays
x_train, y_train = dataset_to_numpy(train_dataset)
x_val, y_val = dataset_to_numpy(validation_dataset)
x_test, y_test = dataset_to_numpy(test_dataset)

print(f'x_train shape: {x_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'x_val shape: {x_val.shape}')
print(f'y_val shape: {y_val.shape}')
print(f'x_test shape: {x_test.shape}')
print(f'y_test shape: {y_test.shape}')




num_pixels = x_train.shape[1] * x_train.shape[2]*x_train.shape[3]
x_train = x_train.reshape(x_train.shape[0], num_pixels)
x_test = x_test.reshape(x_test.shape[0], num_pixels)

#Converti en float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalisation des images
x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

num_classes = y_train.shape[1]

#Création du réseau de neurone
model = neural_network()
model.summary()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25, batch_size=100)#98.36%

#affichage de l'évaluation du model
scores = model.evaluate(x_test, y_test)
print("Neural network accuracy: %.2f%%" % (scores[1]*100)) 

#sauvegarde du model 
model.save(sys.argv[2]+'.h5')

#Sauvegarde du model sous la forme de d'architecture/poids
"""
#sauvegarde dans l'architecture
model_json = model.to_json()
with open(sys.argv[2]+'.json', 'w') as json_file:
    json_file.write(model_json)

# Sauvegarder les poids
model.save_weights(sys.argv[2]+'-poids.h5')
"""
print(sys.argv[2] + " sauvegardé")

