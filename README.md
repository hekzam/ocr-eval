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
Suivez ensuite les instructions d'installation disponible à [cette adresse](https://github.com/tesseract-ocr/tesseract). Veillez à bien cloner le dépot et installer from source pour avoir la version ``

### SimpleHTR

Placez vous dans le répertoire `src/simple-htr`.
Suivez ensuite les instructions d'installation disponible à [cette adresse](https://github.com/githubharald/SimpleHTR). Choisissez le modèle : [Model trained on word images](https://www.dropbox.com/s/mya8hw6jyzqm0a3/word-model.zip?dl=1) qui fonctionne mieux que l'autre dans notre contexte.

**/!\ Attention** : L'installation de SimpleHTR nécéssite l'utilisation d'un environnement virtuel python (venv). Le projet n'étant pas maintenu, vous devez impérativement utiliser une version de python 3.8.x pour pouvoir installer TensorFlow 2.4.0 dans votre environnement. (Vous pouvez également utiliser python 3.9.x, dans ce cas pensez à changer la version de TensorFlow dans le fichier `requirements.txt` en 2.5.0)

Une fois que vous avez suivi les instructions d'installation, copiez le fichier `main.py` présent au départ dans le répertoire, et remplacez le `main.py` contenu dans le répertoire `src/simple-htr/SimpleHTR/src`.

Vous pouvez maintenant vous rendre dans la section **Utilisation des programmes** pour utiliser le programme SimpleHTR modifié.

### Keras
Suivre les instructions d'installation de [cette adresse](https://keras.io/getting_started/) ou la commande `pip install --upgrade keras`

**/!\ Attention** Keras est dépendant de Tensorflow il faut l'avoir installé sur votre machine pour utiliser Keras, [lien des instructions d'installation](https://www.tensorflow.org/install?hl=fr)

## Utilisation des programmes

Les programmes Tesseract, SimpleHTR et Keras prennent 3 paramètres en entrée au minimum, à savoir :
- fichier *path* au format .txt contenant la liste des chemins d'accès aux images du banc de test
- fichier *labels* au format .txt contenant la liste des étiquettes (labels) associés aux images du fichier *path*
- chemin d'accès du fichier de sortie (sortie de type csv, **fortement conseillé** de mettre le nom du fichier en .csv)

### Preprocessing

### Tesseract

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

Le programme prend 3 parametre obligatoire et 1 optionel : 
**`python keras_outcsv.py <fichier liste des chemins> <fichier des labels> <nom fichier csv> <(optionel) nom de model>`**

Les arguments : 
- **fichier_liste_des_chemins** : le chemin d'accès vers le fichier contenant l'ensemble des chemins d'accès des images du test
- **fichier_des_labels** : le chemin d'accès vers le fichier contenant l'ensemble des étiquettes (labels) du test
- **nom_fichier_csv** : le chemin d'accès vers le fichier de sortie
- **nom_de_model** : (optionel) le chemin du model qui doit réaliser les prédictions

**nom_de_model** a par défaut un chemin qui peut ne pas être cohérente avec votre arborescence : *model.h5*

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
