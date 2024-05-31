/**
 * License CC-BY-NC-SA
 * Programme développé dans le cadre d'un projet de Bureau d'Etudes, en Licence
 * informatique à l'université Paul Sabatier. Le projet global, porté par
 * M. Poquet, s'appelle Hekzam.
 *
 * @file csv_prediction.cpp
 *
 * @brief Ce programme permets d'analyser un jeu d'images fournies en entrée
 * et fourni en sortie la liste des prédictions effectuées.
 *
 * @author Oscar Gledel, Laura Guillot, Jean Le Coq
 * Contacts:
 *    - oscar.gledel@univ-tlse3.fr
 *    - laura.guillot@univ-tlse3.fr
 *    - jean.le-coq@univ-tlse3.fr
 */

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#define ARGS_ERROR 1 // Sortie erreur d'arguments
#define FILE_ERROR 2 // Sortie erreur fichier
#define MIN_ARGS 4 // Nombre minimum d'arguments attendus par le programme
#define MAX_ARGS 6 // Nombre maximum d'arguments attendus par le programme
#define DATA_SET_SIZE 60000 // Taille du dataset
#define NB_MAX_PREDICTIONS 25 // Nombre max de predicitons par image

/* On initialise l'API comme variable globale puisqu'elle n'a besoin d'être
initialisée qu'une seule fois */
tesseract::TessBaseAPI *API = new tesseract::TessBaseAPI();

/*============================================================================*/
                /* SECTION POUR LES FONCTIONS AUXILIAIRES */
/*============================================================================*/

/*
 * Renvoi le nom du mode de segmentation sous forme de chaîne de caractères
 */
std::string ToString(tesseract::PageSegMode psm) {
  switch (psm) {
    case tesseract::PSM_OSD_ONLY : // 0
      return "PSM_OSD_ONLY";
    case tesseract::PSM_AUTO_OSD : // 1
      return "PSM_AUTO_OSD";
    case tesseract::PSM_AUTO_ONLY: // 2
      return "PSM_AUTO_ONLY";
    case tesseract::PSM_AUTO: // 3
      return "PSM_AUTO";
    case tesseract::PSM_SINGLE_COLUMN: // 4
      return "PSM_SINGLE_COLUMN";
    case tesseract::PSM_SINGLE_BLOCK_VERT_TEXT: // 5
      return "PSM_SINGLE_BLOCK_VERT_TEXT";
    case tesseract::PSM_SINGLE_BLOCK: // 6
      return "PSM_SINGLE_BLOCK";
    case tesseract::PSM_SINGLE_LINE: // 7
      return "PSM_SINGLE_LINE";
    case tesseract::PSM_SINGLE_WORD: // 8
      return "PSM_SINGLE_WORD";
    case tesseract::PSM_CIRCLE_WORD: // 9
      return "PSM_CIRCLE_WORD";
    case tesseract::PSM_SINGLE_CHAR: // 10
      return "PSM_SINGLE_CHAR";
    case tesseract::PSM_SPARSE_TEXT : // 11
      return "PSM_SPARSE_TEXT";
    case tesseract::PSM_SPARSE_TEXT_OSD : // 12
      return "PSM_SPARSE_TEXT_OSD";
    case tesseract::PSM_RAW_LINE : // 13
      return "PSM_RAW_LINE";
    default :
      return "ERROR";
  }
}

/*
 * Renvoi le mode de segmentation typé par Tesseract à partir d'un entier passé
 * en paramètre. Si l'ID n'existe pas, l'exception n'est pas gérée et le
 * programme fait un segfault
 */
tesseract::PageSegMode psmSelect(int psm_ID) {
  tesseract::PageSegMode psm_list[14] = {
    tesseract::PSM_OSD_ONLY, // 0
    tesseract::PSM_AUTO_OSD, // 1
    tesseract::PSM_AUTO_ONLY, // 2
    tesseract::PSM_AUTO, // 3
    tesseract::PSM_SINGLE_COLUMN, // 4
    tesseract::PSM_SINGLE_BLOCK_VERT_TEXT, // 5
    tesseract::PSM_SINGLE_BLOCK, // 6
    tesseract::PSM_SINGLE_LINE, // 7
    tesseract::PSM_SINGLE_WORD, // 8
    tesseract::PSM_CIRCLE_WORD, // 9
    tesseract::PSM_SINGLE_CHAR, // 10
    tesseract::PSM_SPARSE_TEXT, // 11
    tesseract::PSM_SPARSE_TEXT_OSD, // 12
    tesseract::PSM_RAW_LINE, // 13
  };

  return psm_list[psm_ID];
}

/*
 * Renvoi un entier correspondant au chiffre en entrée. Si la valeur n'est pas
 * un chiffre, renvoi -1.
 */
int charToInt(const char * c) {
  int val = *c - '0';
  if(val <= 9 && val >= 0) {return val;}
  else {return -1;}
}

/*
 * Importe dans un tableau l'ensemble des étiquettes du dataset passé en
 * argument du programme.
 */
void labels_import(const char * labels_path, std::vector<int> * labels_list) {
  std::ifstream labels;
  labels.open(labels_path);

  if(labels.is_open()) {
    int current;
    int i = 0;

    while(labels.good()) {
      labels >> current;
      labels_list->push_back(current);
      i++;
    }

    printf("-> Extraction des labels terminée. Nombre de labels: %d\n", i);
  } else {
    fprintf(stderr, "Could not open file %s\n", labels_path);
    exit(1);
  }
}

/*
 * Initialise Tesseract (/!\ obligatoire au debut du programme /!\)
 * Récupère également les labels dans le fichier de labels
 */
void init(const char * lang, const char * labels_path, std::vector<int> * labels_list) {

  if (API->Init(NULL, lang)) {
    fprintf(stderr, "Could not initialize tesseract.\n");
    exit(1);
  }

  labels_import(labels_path, labels_list);
}


/*============================================================================*/
            /* SECTION POUR LES FONCTIONS LIEES A LA SORTIE CSV */
/*============================================================================*/

/*
 * Ecrit les en têtes du fichier CSV de sortie.
 * /!\ ATTENTION : Le fichier CSV doit déjà être ouvert, on ne fait pas de
 * vérification dans cette fonction.
 */
void writeCSVHeader(std::ofstream& fileStream) {

  fileStream << "Expected value";
  for(int i = 0; i < NB_MAX_PREDICTIONS; ++i) {
    fileStream << ',' << i;
  }
  fileStream << std::endl;
}

/*
 * Ecrit une ligne correspondant à un scan d'image dans le fichier CSV de sortie
 * /!\ ATTENTION : Le fichier CSV doit déjà être ouvert, on ne fait pas de
 * vérification dans cette fonction.
 */
void writeCSVLine(std::ofstream& fileStream, std::vector<const char*> lineData)
{
  int loopCap = NB_MAX_PREDICTIONS;
  if(NB_MAX_PREDICTIONS + 1 > lineData.size()) {loopCap = lineData.size();}
  // + 1 pour la valeur attendu au début
  for(int i = 0; i < loopCap ; ++i) {
    // Cas particulier pour les fichiers CSV
    if(lineData[i][0] == '"') {
      fileStream << "\"\"\"\"" << ',';
    } else {
      fileStream << '"' << lineData[i] << '"' << ',';
    }
  }
  fileStream << std::endl;
}


/*============================================================================*/
              /* SECTION POUR LA FONCTION PRINCIPALE DU PROGRAMME */
/*============================================================================*/

void singleImageScan(const char * image_path, std::string lang,
  const int expected, const tesseract::PageSegMode psm,
  std::ofstream& fileStream)
{
  // Decalarations
  std::vector<const char*> options = {std::to_string(expected).c_str()};

  // Open input image with leptonica library
  Pix *image = pixRead(image_path);
  API->SetImage(image);

  // Réglage de tesseract pour la reconnaissance
  API->SetVariable("lstm_choice_mode", "2");
  // API->SetVariable("tessedit_char_whitelist", "0123456789");
  API->SetPageSegMode(psm);

  // Reconnaissance du caractère dans l'image
  API->Recognize(0);
  tesseract::PageIteratorLevel level = tesseract::RIL_SYMBOL;
  tesseract::ResultIterator* res_it = API->GetIterator();

  // Get confidence level for alternative symbol choices. Code is based on
  // https://github.com/tesseract-ocr/tesseract/blob/a7a729f6c315e751764b72ea945da961638effc5/src/API/hocrrenderer.cpp#L325-L344
  std::vector<std::vector<std::pair<const char*, float>>>* choiceMap = nullptr;
  if (res_it != 0) {
    do {
      const char* word;
      int x1, y1, x2, y2;
      res_it->BoundingBox(level, &x1, &y1, &x2, &y2);
      choiceMap = res_it->GetBestLSTMSymbolChoices();
      // Si on a bien reconnu des chaînes de caractères
      if(choiceMap != NULL) {
        // Pour chaque chaine de caractère reconnue
        for (auto timestep : *choiceMap) {
          // Si il y a au moins une option trouvée pour cette chaine
          if (timestep.size() > 0) {
            // Pour chaque option de la chaine
            for (auto & j : timestep) {
              word =  j.first;
              options.push_back(word);
            }
          }
        }
      } else {
        printf("Error : Skipped image %s\n", image_path);
      }
    } while (res_it->Next(level));
  }

  // Destroy image after usage
  pixDestroy(&image);

  // Ecriture dans le CSV
  writeCSVLine(fileStream, options);
}


/*============================================================================*/
                            /* SECTION POUR LE MAIN */
/*============================================================================*/

int main(int argc, char* argv[])
{
  /* ########################## ARGS CHECK SECTION ########################## */

  // Vérification de la quantité d'arguments
  if(argc < MIN_ARGS || argc > MAX_ARGS) {
    printf("usage : %s image_path labels_path output_file [psm_mode"
      "(default = 10)] [traineddata_file (default = fra)]\n", argv[0]);
    return ARGS_ERROR;
  }

  /* ==== ATTRIBUTION DES ARGUMENTS === */
  // Arguments obligatoires
  std::string im_path = argv[1];
  std::string lb_path = argv[2];
  std::string out_path = argv[3];

  // Arguments optionnels
  auto psm_mode = tesseract::PSM_SINGLE_CHAR;
  if(argc > 4) {psm_mode = psmSelect(std::stoi(argv[4]));}
  std::string lang = "fra";
  if(argc > 5) {lang = argv[5];}

  printf("-> Correspondance des arguments: OK\n");
  printf("\t-> Images : \t%s\n", im_path.c_str());
  printf("\t-> Labels : \t%s\n", lb_path.c_str());
  printf("\t-> Ouput : \t%s\n", out_path.c_str());
  printf("\t-> PSM : \t%s\n", ToString(psm_mode).c_str());
  printf("\t-> Model : \t%s\n", lang.c_str());

  /* ############################# INIT SECTION ############################# */

  std::vector<int> labels;
  std::ifstream path_list;
  std::string current_path;
  init(lang.c_str(), lb_path.c_str(), &labels);

  // Check labels file
  if(im_path.find(".txt") == std::string::npos) {
    fprintf(stderr, "File ''%s' is not a txt file.\n", im_path.c_str());
    return ARGS_ERROR;
  }

  // Check paths file
  path_list.open(im_path);
  if(!path_list.is_open()) {
    fprintf(stderr, "Could not open file ''%s'.\n", im_path.c_str());
  }

  printf("-> Lecture des fichiers d'entrée : OK\n");

  /* ######################### OUTPUT CONFIG SECTION ######################## */

  std::ofstream outputFile(out_path);
  if(!outputFile) {
    std::cerr << "Error during ouput file creation." << std::endl;
    return FILE_ERROR;
  }

  writeCSVHeader(outputFile);

  printf("-> Creation du fichier de sortie : OK\n");

  /* ########################### MAIN LOOP SECTION ########################## */

  int i = 0;
  while(path_list.good()) {
    path_list >> current_path;
    singleImageScan(current_path.c_str(), lang, labels[i], psm_mode, outputFile);
    i++;
  }

  printf("-> Prédiction sur les images d'entrée : OK\n");

  /* ######################## MEMORY RELEASE SECTION ######################## */

  // Destroy used object and release memory
  API->End();
  delete API;

  printf("-> Libération des espaces mémoires et de Tesseract : OK\n");

  return 0;
}
