# README - OCR Eval - BE 2024

## A propos

Ce dépôt contient l'ensemble des fichiers sources créés dans le cadre de l'UE B.E. pour le projet de reconnaissance de caractères inclus dans Hekzam. Les procédure d'installation et de lancement des différents programmes sont disponibles plus loin dans ce fichier.

**Contributeurs :**
- Laura Guillot
- Jean Le Coq
- Oscar Gledel

## Organisation du dépôt

- **src**
	- **tesseract** : sources utiles pour l'automatisation de Tesseract (c++)
	- **simple-htr** : sources utiles pour l'automatisation de SimpleHTR (python)
	- **keras** : sources utiles pour l'automatisation de Keras (python)
	- **csv** : sources utilisées pour obtenir les graphiques à partir des fichiers csv
   	- **prep** : sources pour le preprocessing des fichiers/des images
- **resources** : dossiers d'images et fichiers de paths/labels
	- **custom** : images issues des données manuscrites remplies par les contributeurs
	- **mnist** : images issues du jeu de donnée MNIST de Yann Le Cun
- **results** : fichiers de sorties des différents programmes
	- **r_csv** : fichiers .csv contenant les résultats des modèles
	- **img** : graphiques générés à partir des fichiers .csv (format .png)

## Procédures d'installation

Commencez par cloner ce dépôt dans le répertoire de votre choix avec `git clone https://github.com/hekzam/ocr-eval.git`, puis rendez vous dans le dépôt avec `cd ocr-eval`. Vous pouvez ensuite suivre les procédures d'installation spécifiques aux modèles en fonction de ceux que vous souhaitez utiliser.

### Tesseract

Placez vous dans le répertoire `src/tesseract`.
Suivez ensuite les instructions d'installation disponible à [cette adresse](https://github.com/tesseract-ocr/tesseract). Veillez à bien cloner le dépot et installer from source pour avoir la version `tesseract 5.3.4-49-g577e8` (en réalité il n'est pas nécéssaire d'avoir cette version spécifique, même si cela limitera le risque de bugs. Vous pouvez installer une autre version du moment qu'elle est en `5.x.x`).

Une fois l'installation terminée, allez dans le répertoire `src/tesseract/tesseract/tessdata` et copiez y les fichiers **.traineddata** contenu dans le répertoire `src/tesseract`. Enfin, revenez dans ce répertoire.

Enfin, compilez le programme avec la commande suivante : **`g++ ocr_prediction.cpp -o ocr_prediction pkg-config --cflags --libs tesseract lept`**. Les dépendances sont normalement installées lors de l'installation de tesseract, mais au besoin vous pouvez les installer manuellement avec apt par exemple. En entrant la commande `tesseract --version` vous devriez avoir quelque chose de similaire à ça :

```
tesseract 5.3.4-49-g577e8
 leptonica-1.82.0
  libgif 5.2.1 : libjpeg 6b (libjpeg-turbo 2.1.2) : libpng 1.6.39 : libtiff 4.5.0 : zlib 1.2.13 : libwebp 1.2.4 : libopenjp2 2.5.0
 Found AVX2
 Found AVX
 Found FMA
 Found SSE4.1
 Found OpenMP 201511
 Found libarchive 3.6.2 zlib/1.2.13 liblzma/5.4.0 bz2lib/1.0.8 liblz4/1.9.4 libzstd/1.5.2
 Found libcurl/7.88.1 NSS/3.87.1 zlib/1.2.13 brotli/1.0.9 zstd/1.5.4 libidn2/2.3.3 libpsl/0.21.2 (+libidn2/2.3.3) libssh2/1.10.0 nghttp2/1.52.0 librtmp/2.3 OpenLDAP/2.5.13
```

Vous pouvez maintenant vous rendre dans la section **Utilisation des programmes** pour utiliser le programme tesseract modifié.

### SimpleHTR

Placez vous dans le répertoire `src/simple-htr`.
Suivez ensuite les instructions d'installation disponible à [cette adresse](https://github.com/githubharald/SimpleHTR). Choisissez le modèle : [Model trained on word images](https://www.dropbox.com/s/mya8hw6jyzqm0a3/word-model.zip?dl=1) qui fonctionne mieux que l'autre dans notre contexte.

**/!\ Attention** : L'installation de SimpleHTR nécéssite l'utilisation d'un environnement virtuel python (venv). Le projet n'étant pas maintenu, vous devez impérativement utiliser une version de python 3.8.x pour pouvoir installer TensorFlow 2.4.0 dans votre environnement. (Vous pouvez également utiliser python 3.9.x, dans ce cas pensez à changer la version de TensorFlow dans le fichier `requirements.txt` en 2.5.0)

Une fois que vous avez suivi les instructions d'installation, copiez le fichier `main.py` présent au départ dans le répertoire, et remplacez le `main.py` contenu dans le répertoire `src/simple-htr/SimpleHTR/src`.

Vous pouvez maintenant vous rendre dans la section **Utilisation des programmes** pour utiliser le programme SimpleHTR modifié.

### Keras
Suivre les instructions d'installation de [cette adresse](https://keras.io/getting_started/) ou la commande `pip install --upgrade keras`

**/!\ Attention** Keras est dépendant de Tensorflow il faut l'avoir installé sur votre machine pour utiliser Keras, [lien des instructions d'installation](https://www.tensorflow.org/install?hl=fr)

Vous pouvez maintenant vous rendre dans la section **Utilisation des programmes** pour utiliser le programme Keras modifié.

## Utilisation des programmes

Les programmes Tesseract, SimpleHTR et Keras prennent 3 paramètres en entrée au minimum, à savoir :
- fichier *path* au format .txt contenant la liste des chemins d'accès aux images du banc de test
- fichier *labels* au format .txt contenant la liste des étiquettes (labels) associés aux images du fichier *path*
- chemin d'accès du fichier de sortie (sortie de type csv, **fortement conseillé** de mettre le nom du fichier en .csv)

### Preprocessing

Il y a deux programmes de pre-processing dans le répertoire `src/prep`.

Le premier, `preprocessing.py`, est fixe et a permi de créer le jeu de données personnalisées que vous pouvez trouver dans le répertoire `resources/custom`. L'intérêt premier de ce programme est de pouvoir le consulter pour connaître les réglages et les fonctions de la librairie **OpenCV** utilisées.

Le second, `generate_paths.py`, permet de générer le fichier des chemins d'accès des images dynamiquement. Voici l'affichage de l'aide (avec la commande `python generate_paths.py -h`) ainsi qu'un détail sur les arguments :
```
usage: generate_paths.py [-h] [--path PATH] --output OUTPUT [--mode {custom,mnist}]

options:
  -h, --help            show this help message and exit
  --path PATH           Prefix path to add to filenames
  --output OUTPUT       Output file for paths.
  --mode {custom,mnist}
```
- **path** : éventuel arborescence d'accès aux images. Par défaut, si rien n'est spécifié, l'arborescence sera `nom_jeu_donnee/nom_image.png` pour chaque image.
- **output** : nom du fichier de sortie.
- **mode** : jeu de données sur lequel se baser pour formuler les noms des images .Par défaut *custom*.

### Tesseract

Ce programme prends 3 paramètres obligatoires et 2 optionnels pour fonctionner. Placez vous dans le répertoire `src/tesseract` puis exécutez la commande suivante :

**`./csv_prediction paths_file labels_file csv_output <psm_mode> <model>`**

Arguments obligatoires :
- **paths_file** : le chemin d'accès vers le fichier contenant l'ensemble des chemins d'accès des images du test
- **labels_file** : le chemin d'accès vers le fichier contenant l'ensemble des étiquettes (labels) du test
- **csv_output** : le chemin d'accès vers le fichier de sortie

Arguments optionnels :
- **psm_mode** : mode de segmentation des pages de tesseract (par défaut mode 10). Pour un détail des modes exécutez la commande `tesseract --help-psm`.
- **model** : modèle de reconnaissance utilisé (par défaut 'fra'), des modèles entrainé en fine-tuning pendant le B.E sont disponibles dans le dossier `src/tesseract` au format *.traineddata*.

### SimpleHTR

Ce programme prends 4 paramètres obligatoires (dont les 3 paramètres cités en introduction de cette partie) pour fonctionner. Placez vous dans le répertoire `src/simple-htr/SimpleHTR/src` puis exécutez la commande suivante :

**`python main.py --mode stats --paths paths_file --labels labels_file -- csv_out csv_file`**

Les arguments sont interchangeables car le programme utilise la librairie *argparse* pour les gérer. Voyons le détail des arguments :
- **mode** : mode utilisé, le programme fait un switch en fonction du mode et exécute des instructions en fonction. Ce mode est un mode custom et représente le principal changement avec le programme d'origine
- **paths** : le chemin d'accès vers le fichier contenant l'ensemble des chemins d'accès des images du test
- **labels** : le chemin d'accès vers le fichier contenant l'ensemble des étiquettes (labels) du test
- **csv_out** : le chemin d'accès vers le fichier de sortie

Les arguments ont des valeurs par défaut qui peuvent ne pas être cohérentes avec votre arborescence. **Par sécurité**, renseignez une valeur pour l'ensemble de ces arguments.

### Keras

Pour entraîner un modele keras : 
Le programme prend 3 paramètre obligatoire et 1 optionel : 
**`python keras_outcsv.py <fichier liste des chemins> <fichier des labels> <nom fichier csv> <(optionnel) nom de modele default = model.h5>`**

Les arguments : 
- **fichier_liste_des_chemins** : le chemin d'accès vers le fichier contenant l'ensemble des chemins d'accès des images du test
- **fichier_des_labels** : le chemin d'accès vers le fichier contenant l'ensemble des étiquettes (labels) du test
- **nom_fichier_csv** : le chemin d'accès vers le fichier de sortie
- **nom_de_model** : (optionel) le chemin du modele qui doit réaliser les prédictions

**nom_de_model** a par défaut un chemin qui peut ne pas être cohérent avec votre arborescence : *model.h5*

Pour créer un modèle keras : 
Le programme 'creation_model prend 2 paramètre obligatoire et 2 parametre optionnel : 
**`python keras_outcsv.py <chemin du repertoire d'entrainement>  <nom du model > <(optionnel) taille x default = 47>  <(optionnel) taille y default = 63>`**
Les arguments : 
- **chemin du repertoire d'entrainement** : le chemin du répertoire utilisé pour entrainer le model, le répertoire doit être organisé et le élèment trier par classe comme ceci : repertoire/classeA/...
  																							    /classeB/...
- **nom du model** : designe le nom du modele keras
- **taille x** : (optionel) taille en largeur de l'image, par default la valeur 47 qui represente la largeur des images du dataset custom
- **taille y** : (optionel) taille en longeur de l'image, par default la valeur 63 qui represente la longueur des images du dataset custom

### Generate Plots

La version de python utilisée ici pour le programme est la `3.11.2`, mais la version de python ne devrait pas poser de problème si elle est équivalente ou plus récente.
Voici les dépendances de ce programme :
```python
import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
```
Ce programme propose une option d'aide en entrant la commande `python generate_plots.py -h`. Voici ce qu'affiche la commande et un détail des différentes options ci-dessous.

```
usage: generate_plots.py [-h] --file FILE --output OUTPUT [--mode {confusion,distribution,report,roc,norm_confusion,all}] [--title TITLE]
                         [--color {Reds,Blues,Greens,Yellows}] [--filtered]

options:
  -h, --help            show this help message and exit
  --file FILE           CSV file to plot.
  --output OUTPUT       Output file path.
  --mode {confusion,distribution,report,roc,norm_confusion,all}
  --title TITLE         Title of the plot to create
  --color {Reds,Blues,Greens,Yellows}
  --filtered            Filter non-digits values
```

Les arguments obligatoires sont :
- `--file` fichier source (attendu au format .csv ou équivalent) à traiter pour produire le plot sélectionné
- `--output` fichier OU nom de répertoire pour le(s) fichier(s) en sortie

Les arguments optionnels sont :
- `--mode` choix possible parmi les suivants
	- *confusion* : matrice de confusion
   	- *distribution* : histogramme de distribution des valeurs attendues et prédites
   	- *report* : rapport de classification contenant la précision, le rappel, et le F1 score
   	- *roc* : courbe ROC et calcul de l'AUC pour chaque classe prédite (peu intéréssant dans le cas des modèles actuels)
   	- *norm_confusion* : matrice de confusion normalisée (colorimétrie identique mais valeurs plus 'lisibles')
   	- *all* : produit l'ensemble des plots cités plus haut. Attention, dans ce mode le paramètre `--output` est utilisé comme nom de dossier !
- `--title` titre du graphique (inutile en cas de choix du mode all)
- `--color` nuancier de couleur (pour les plots qui acceptent les variations de couleurs)
- `--filtered` données filtrées ou non. Dans tout les cas, les données sont triées comme un chiffre ou comme 'other'. Si le mode filtered est activé, on choisira le premier chiffre proposé pour chaque prédiction, sinon on prendra 'other'
