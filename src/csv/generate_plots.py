import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


# Variables globales : utilisées comme des #define en C ou C++
PREDICTED = '0'
EXPECTED = 'Expected value'


################################################################################
#                      SECTION DES FONCTIONS DE FILTRAGE                       #
################################################################################

def erase_no_digits(values_list):
    """Filtre les labels pour garder uniquement les chiffres de 0 à 9 et groupe
    les autres dans une categorie 'other'"""

    def categorize_label(label):
        """Inner fonction servant de lambda"""
        if label.isdigit() and 0 <= int(label) <= 9:
            return label
        else:
            return 'other'

    values_list = values_list.apply(categorize_label)
    return values_list


def first_int_prediction(data):
    """Renvoi une liste des predictions en selectionnant le chiffre avec le
    plus de confiance"""
    filtered_predictions = []
    for index, row in data.iterrows():
        found = False
        for pred in row[1:]:
            if pred.isdigit() :
                filtered_predictions.append(pred)
                found = True
                break
        if not found:
            filtered_predictions.append('other')

    return filtered_predictions


################################################################################
#                     SECTION DES FONCTIONS SUR LES FICHIERS                   #
################################################################################

def load_data(file_path):
    """Charge le fichier CSV passé en paramètre"""

    data = pd.read_csv(file_path)
    data = data.astype(str)
    print(f"Fichier {file_path} correctement chargé.")
    return data


def check_file_integrity(data):

    if EXPECTED not in data.columns or PREDICTED not in data.columns:
        print("CSV file must contain 'Expected value' and '0' columns.")
        sys.exit(1)

    return


################################################################################
#                       SECTION DES FONCTIONS DE PLOTS                         #
################################################################################

def plot_confusion_matrix(y_true, y_pred, output_file,
    title='Confusion Matrix', color='Blues'):
    """Génère une matrice de confusion et la sauvegarde dans le dossier
    courant"""

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap=color)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig(output_file)
    plt.close()

    print(f"Matrice {title} sauvegardée comme {output_file}")


def plot_distribution(y_true, y_pred, output_file):
    """Génère un histogramme de distribution des valeurs prédictives et
    attendues."""
    plt.figure(figsize=(14, 7))

    # Plot distribution of true values
    plt.subplot(1, 2, 1)
    sns.histplot(y_true, bins=11, kde=False, color='blue')
    plt.title('Distribution des valeurs attendues')
    plt.xlabel('Valeur')
    plt.ylabel('Fréquence')

    # Plot distribution of predicted values
    plt.subplot(1, 2, 2)
    sns.histplot(y_pred, bins=11, kde=False, color='orange')
    plt.title('Distribution des valeurs prédictives')
    plt.xlabel('Valeur')
    plt.ylabel('Fréquence')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"Distribution des valeurs sauvegardée comme {output_file}")


def plot_classification_report(y_true, y_pred, output_file, color='Blues'):
    """Génère un rapport de classification sous forme de graphique."""
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    plt.figure(figsize=(10, 5))
    sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap=color)
    plt.title('Rapport de Classification')
    plt.savefig(output_file)
    plt.close()

    print(f"Rapport de classification sauvegardé comme {output_file}")


def plot_roc_auc(y_true, y_pred, output_file):
    """Génère les courbes ROC et calcule l'AUC pour chaque classe."""
    classes = sorted(list(set(y_true) | set(y_pred)))
    y_true_bin = label_binarize(y_true, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], label=f'Classe {classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbes ROC par classe')
    plt.legend(loc="lower right")
    plt.savefig(output_file)
    plt.close()

    print(f"Courbes ROC sauvegardées comme {output_file}")


def plot_normalized_confusion_matrix(y_true, y_pred, output_file,
    title='Matrice de Confusion Normalisée', color='Blues'):
    """Génère une matrice de confusion normalisée et la sauvegarde."""
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=color)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig(output_file)
    plt.close()

    print(f"Matrice {title} sauvegardée comme {output_file}")


def plot_all(y_true, y_pred, output_folder, color='Blues'):

    os.mkdir(output_folder)

    plot_confusion_matrix(y_true, y_pred, output_folder + '/confusion.png',
        title='Confusion Matrix', color=color)
    plot_distribution(y_true, y_pred, output_folder + '/distribution.png')
    plot_classification_report(y_true, y_pred,
        output_folder + '/classification.png', color=color)
    plot_roc_auc(y_true, y_pred, output_folder + '/roc_auc.png')
    plot_normalized_confusion_matrix(y_true, y_pred,
        output_folder + '/norm_confusion.png', color=color)


################################################################################
#                    SECTION DES FONCTIONS SUR LES ARGUMENTS                   #
################################################################################

def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--file', required=True, type=str,
        help='CSV file to plot.')

    parser.add_argument('--output', required=True, type=str,
        help='Output file path.')

    parser.add_argument('--mode',
        choices=['confusion', 'distribution', 'report', 'roc',
                 'norm_confusion', 'all'],
        default='confusion')

    parser.add_argument('--title', type=str, help='Title of the plot to create')

    parser.add_argument('--color',
        choices=['Reds', 'Blues', 'Greens', 'Yellows'],
        default='Blues')

    parser.add_argument('--filtered', action='store_true',
        help='Filter non-digits values')

    return parser.parse_args()


################################################################################
#                                SECTION MAIN                                  #
################################################################################

def main():

    args = parse_args()

    file_path = args.file
    data = load_data(file_path)
    check_file_integrity(data)

    y_true = data[EXPECTED]
    if args.filtered:
        y_pred = first_int_prediction(data)
    else:
        y_pred = erase_no_digits(data[PREDICTED])


    if args.mode == 'confusion':
        plot_confusion_matrix(y_true, y_pred,
            args.output, color=args.color)

    elif args.mode == 'distribution':
        plot_distribution(y_true, y_pred, args.output)

    elif args.mode == 'report':
        plot_classification_report(y_true, y_pred, args.output,
            color=args.color)

    elif args.mode == 'roc':
        plot_roc_auc(y_true, y_pred, args.output)

    elif args.mode == 'norm_confusion':
        plot_normalized_confusion_matrix(y_true, y_pred, args.output,
            color=args.color)

    elif args.mode == 'all':
        plot_all(y_true, y_pred, args.output, color=args.color)


if __name__ == "__main__":
    main()
