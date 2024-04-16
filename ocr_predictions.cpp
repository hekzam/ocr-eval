#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include <string>
#include <iostream>
#include <fstream>

#define OCR_DIRECT 0 // La valeur attendue a directement été trouvée
#define OCR_FIRST 1 // La valeur attendue est le premier chiffre de la liste des GetBestLSTMSymbolChoices
#define OCR_INLIST 2 // La valeur attendue est dans la liste mais n'est pas le premier nombre
#define OCR_OUTLIST 3 // La valeur attendue n'est pas dans la liste
#define OCR_NAN 4 // Aucune valeur n'as été trouvée pour cette image
#define RETURN_TYPES 5 // Nombre de types de retour pour le tableau

#define ARGS_ERROR 1
#define MIN_ARGS 4
#define MAX_ARGS 6
#define DATA_SET_SIZE 60000
#define TEST_CAP 50 // Pour les tests sur MNIST, permets de caper le nombre de
                    // tests

/* On initialise l'API comme variable globale puisqu'elle n'a besoin d'être
initialisée qu'une seule fois */
tesseract::TessBaseAPI *API = new tesseract::TessBaseAPI();


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

int charToInt(const char * c) {
  int val = *c - '0';
  if(val <= 9 && val >= 0) {return val;}
  else {return -1;}
}

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

/* Fonction principale : scanne l'image et donne les niveaux de confiance pour
   différents caractères */
int singleImageScan(const char * image_path, std::string lang, bool print_mode,
  const int expected, const tesseract::PageSegMode psm)
{
  int test_identifier = OCR_NAN;

  // Open input image with leptonica library
  Pix *image = pixRead(image_path);
  API->SetImage(image);

  // Réglage de tesseract pour la reconnaissance
  API->SetVariable("lstm_choice_mode", "2");
  API->SetPageSegMode(psm);

  // Affichage d'en tête
  if(print_mode) {
    printf("===========|| Fichier : %s ||==========\n",image_path);
    printf("###  RECONNU COMME :   %s", API->GetUTF8Text());
    printf("###  VALEUR ATTENDUE : %d\n\n", expected);
  }

  // Reconnaissance du caractère dans l'image
  API->Recognize(0);
  tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
  tesseract::ResultIterator* res_it = API->GetIterator();

  // Get confidence level for alternative symbol choices. Code is based on
  // https://github.com/tesseract-ocr/tesseract/blob/a7a729f6c315e751764b72ea945da961638effc5/src/API/hocrrenderer.cpp#L325-L344
  std::vector<std::vector<std::pair<const char*, float>>>* choiceMap = nullptr;
  if (res_it != 0) {
    do {
      const char* word;
      float conf;
      int x1, y1, x2, y2, tcnt = 1, gcnt = 1, wcnt = 0;
      res_it->BoundingBox(level, &x1, &y1, &x2, &y2);
      choiceMap = res_it->GetBestLSTMSymbolChoices();
      // Segmentation fault à la ligne en dessous (probablement à l'assignation)
      if(choiceMap != NULL) {
        // Dans le cas où test_identifier n'a pas encore été mis à jour
        if(test_identifier == OCR_NAN) {test_identifier = OCR_OUTLIST;}

        for (auto timestep : *choiceMap) {
          if(print_mode) {printf("=> Caractère n°%d : reconnu comme %s\n",
            wcnt, timestep[0].first);}

          if (timestep.size() > 0) {
            bool intPassed = false;
            int option_ID = 1;
            for (auto & j : timestep) {
              conf = 100.0 - j.second;
              word =  j.first;
              if(print_mode) {printf("\toption %d: '%s'  \tconfiance: %.2f %\n",
                        option_ID, word, conf);}
              gcnt++;
              option_ID++;

              // Vérification pour la valeur de retour
              int firstInt = charToInt(j.first);
              if(firstInt != -1) {
                if(firstInt == expected && !intPassed)
                  {test_identifier = OCR_FIRST; intPassed = true;}
                else if (firstInt == expected && intPassed)
                  {test_identifier = OCR_INLIST;}
                else
                  {intPassed = true;}
              }
            }
            tcnt++;
          }
          wcnt++;
          if(print_mode) {printf("\n");}
        }
      } else {printf("Segmentation fault skipped at image %s\n", image_path);}
    } while (res_it->Next(level));
    if(print_mode) {printf("=====================================================\n\n");}
  }

  auto estimated = API->GetUTF8Text();
  estimated[std::strlen(estimated) - 1] = 0; // Retirer le \n
  if(std::strcmp(estimated, std::to_string(expected).c_str()) == 0) {test_identifier = OCR_DIRECT;}

  // Destroy image after usage
  pixDestroy(&image);
  return test_identifier;
}

// Importe dans un tableau l'ensemble des étiquettes du dataset
int * labels_import(const char * labels_path, int * labels_list) {
  std::ifstream labels;
  labels.open(labels_path);

  if(labels.is_open()) {
    std::string current;
    int i = 0;

    while(labels.good()) {
      labels >> current;
      labels_list[i] = std::stoi(current);
      i++;
    }

    printf("-> Extraction des labels terminée. Nombre de labels: %d\n", i+1);
    return labels_list;
  } else {
    fprintf(stderr, "Could not open file %s\n", labels_path);
    return NULL;
  }
}


void init(const char * lang, const char * labels_path, int * labels_list) {
  // Initialize tesseract-ocr with English, without specifying tessdata path
  if (API->Init(NULL, lang)) {
    fprintf(stderr, "Could not initialize tesseract.\n");
    exit(1);
  }

  labels_import(labels_path, labels_list);
}


/** Programme de statistique sur la reconnaissance de caractères
* ARG 1 (obligatoire) : emplacement du fichier à traiter
* ARG 2 (facultatif) : langue utilisée (par défaut : français)
*/
int main(int argc, char* argv[])
{
  /* ########################## ARGS CHECK SECTION ########################## */

  if(argc < MIN_ARGS || argc > MAX_ARGS) {
    printf("usage : ocr_predictions image_path labels_path psm_mode [lang (default = fra)] [mode (detail | stats)(default = on)]\n");
    return ARGS_ERROR;
  }

  std::string im_path = argv[1];
  std::string lb_path = argv[2];
  auto psm_mode = psmSelect(std::stoi(argv[3]));



  auto lang = "fra";
  if(argv[4] != NULL) {lang = argv[4];}

  // En mode stats, on affiche seulement la matrice de confusion
  bool print_mode = true;
  if(argv[5] != NULL) {
    if(!strcmp(argv[5], "detail")){print_mode = true;}
    else if(!strcmp(argv[5], "stats")){print_mode = false;}
    else{
      printf("usage : ocr_predictions image_path labels_path psm_mode [lang (default = fra)] [mode (detail | stats)(default = on)]\n");
      return ARGS_ERROR;
    }
  }

  /* ############################# INIT SECTION ############################# */

  int labels[DATA_SET_SIZE];
  init(lang, lb_path.c_str(), labels);
  int OCR_result[RETURN_TYPES];
  for(int n = 0; n < RETURN_TYPES; n++) {
    OCR_result[n] = 0;
  }

  int i = 0;

  /* ########################### MAIN LOOP SECTION ########################## */

  // If txt, iterate on each line. else = mono-scan
  if(im_path.find(".txt") != std::string::npos) {
    std::ifstream path_list;
    path_list.open(im_path);

    if(path_list.is_open()) {
      std::string current_path;
      while(path_list.good() && i < TEST_CAP) {
        path_list >> current_path;
        int result = singleImageScan(current_path.c_str(), lang, print_mode, labels[i], psm_mode);
        OCR_result[result]++;
        i++;
      }
    }
    else {
      fprintf(stderr, "Could not open file ''%s'.\n", im_path.c_str());
    }
  }
  else {
    int result = singleImageScan(im_path.c_str(), lang, print_mode, labels[0], psm_mode);
    OCR_result[result]++;

  }

  printf("==========|| RESULTATS ||==========\n");
  printf("--> Mode utilisé : %s\n", ToString(psm_mode).c_str());
  printf("-> DIRECT : %d / %d\n", OCR_result[OCR_DIRECT], i);
  printf("-> PREMIER NOMBRE : %d / %d\n", OCR_result[OCR_FIRST], i);
  printf("-> DANS LA LISTE : %d / %d\n", OCR_result[OCR_INLIST], i);
  printf("-> HORS DE LA LISTE : %d / %d\n", OCR_result[OCR_OUTLIST], i);
  printf("-> SCAN IMPOSSIBLE : %d / %d\n", OCR_result[OCR_NAN], i);

  /* ######################## MEMORY RELEASE SECTION ######################## */

  // Destroy used object and release memory
  API->End();
  delete API;
  return 0;
}
