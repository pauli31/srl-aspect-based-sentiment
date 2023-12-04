import os
from pathlib import Path

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_PATH, 'data')
LOG_DIR = os.path.join(BASE_PATH, 'logs')
RESULTS_DIR = os.path.join(BASE_PATH, "results")
WANDB_API_KEY_DIR = os.path.join(BASE_PATH, 'wandb_private','wandbkey.txt')
TRAINED_MODELS_DIR = os.path.join(BASE_PATH, 'trained_models')
TRANSFORMERS_TRAINED_MODELS = os.path.join(TRAINED_MODELS_DIR, 'transformers')
WANDB_DIR = os.path.join(BASE_PATH, 'wandb')

LM_DATA_DIR_TIGI = os.path.join(DATA_DIR, 'lm', 'raw','tigi')
LM_DATA_DIR_TIGI_CS = os.path.join(LM_DATA_DIR_TIGI, 'cs')
LM_DATA_DIR_TIGI_EN = os.path.join(LM_DATA_DIR_TIGI, 'en')


Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(TRANSFORMERS_TRAINED_MODELS).mkdir(parents=True, exist_ok=True)
Path(LM_DATA_DIR_TIGI).mkdir(parents=True, exist_ok=True)

LOGGING_FORMAT = '%(asctime)s: %(levelname)s: %(name)s %(message)s'
LOGGING_DATE_FORMAT = '%m/%d/%Y %H:%M:%S'

DATASET_LANGS = ['cs','en']
# type of datasource
DATASET_SOURCES = ['semeval2014', 'semeval2016', 'sentihood']
DATASET_LIST = ['semeval2014_en', 'semeval2014_cs', 'semeval2014_cs-cross_val','cs_absa_srl_dataset', 'cs_srl_e2e', 'cs_srl' , 'en_absa_srl_dataset']


# cat - aspect category recognition and sentiment classification
# term - aspect term recognition and sentiment classification

TASK_TYPES = ["CAT", "TERM", "PRE-TRAIN","SRL"]
CAT_SOLUTION_TYPES = ["QA_M", "QA_B", "NLI_M", "NLI_B"]

INJECTION_MODE_AVERAGE = 'average'
INJECTION_MODE_CONCAT_AVERAGE = 'concat-avg'
INJECTION_MODE_CONCAT_CONVOLUTION = 'concat-convolution'
INJECTION_MODE_MULTI_TASK = 'multi-task'



INJECTION_MODES = [INJECTION_MODE_AVERAGE, INJECTION_MODE_CONCAT_AVERAGE, INJECTION_MODE_CONCAT_CONVOLUTION, INJECTION_MODE_MULTI_TASK]

SCHEDULERS = ["linear_wrp", "constant", "cosine_wrp", "polynomial_wrp", "constant_warmp_up"]
OPTIMIZERS_CHOICES = ['Adam', 'AdamW', 'RMSprop', 'Adagrad', 'Adadelta', 'SGD']


RAW_DATASETS_DIR = os.path.join(DATA_DIR, 'raw-datasets')
GENERATED_DATASET_DIR = os.path.join(DATA_DIR, 'datasets')
GENERATED_DATASET_DIR_SEMEVAL = os.path.join(GENERATED_DATASET_DIR, 'semeval')
GENERATED_DATASET_DIR_SEMEVAL_PAIR = os.path.join(GENERATED_DATASET_DIR_SEMEVAL, 'pair')
GENERATED_DATASET_DIR_SEMEVAL_PAIR_CS = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_PAIR, 'cs')
GENERATED_DATASET_DIR_SEMEVAL_PAIR_CS_KFOLD = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_PAIR, 'cs', 'kfold')
GENERATED_DATASET_DIR_SEMEVAL_PAIR_EN = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_PAIR, 'en')
GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA = os.path.join(GENERATED_DATASET_DIR_SEMEVAL, 'custom_qa')
GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA_EN = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA, 'en')
GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA_CS = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA, 'cs')
GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA_CS_KFOLD = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA, 'cs', 'kfold')
GENERATED_DATASET_DIR_SEMEVAL_SPLIT_CS = os.path.join(GENERATED_DATASET_DIR_SEMEVAL, 'split', 'cs')
GENERATED_DATASET_DIR_SEMEVAL_SPLIT_CS_KFOLD = os.path.join(GENERATED_DATASET_DIR_SEMEVAL, 'split', 'cs', 'kfold')
GENERATED_DATASET_DIR_SEMEVAL_SPLIT_EN = os.path.join(GENERATED_DATASET_DIR_SEMEVAL, 'split', 'en')

Path(GENERATED_DATASET_DIR).mkdir(parents=True, exist_ok=True)
Path(GENERATED_DATASET_DIR_SEMEVAL).mkdir(parents=True, exist_ok=True)
Path(GENERATED_DATASET_DIR_SEMEVAL_PAIR).mkdir(parents=True, exist_ok=True)
Path(GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA).mkdir(parents=True, exist_ok=True)
Path(GENERATED_DATASET_DIR_SEMEVAL_SPLIT_CS).mkdir(parents=True, exist_ok=True)
Path(GENERATED_DATASET_DIR_SEMEVAL_SPLIT_EN).mkdir(parents=True, exist_ok=True)
Path(GENERATED_DATASET_DIR_SEMEVAL_PAIR_CS).mkdir(parents=True, exist_ok=True)
Path(GENERATED_DATASET_DIR_SEMEVAL_PAIR_EN).mkdir(parents=True, exist_ok=True)
Path(GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA_EN).mkdir(parents=True, exist_ok=True)
Path(GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA_CS).mkdir(parents=True, exist_ok=True)
Path(GENERATED_DATASET_DIR_SEMEVAL_SPLIT_CS_KFOLD).mkdir(parents=True, exist_ok=True)
Path(GENERATED_DATASET_DIR_SEMEVAL_PAIR_CS_KFOLD).mkdir(parents=True, exist_ok=True)
Path(GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA_CS_KFOLD).mkdir(parents=True, exist_ok=True)


EN_SEMEVAL_2014_RAW_DATASET_TRAIN = os.path.join(RAW_DATASETS_DIR, 'english', 'semeval-2014', 'github', 'BERT-AUX', 'Restaurants_Train.xml')
EN_SEMEVAL_2014_RAW_DATASET_TEST = os.path.join(RAW_DATASETS_DIR, 'english', 'semeval-2014', 'github', 'BERT-AUX', 'Restaurants_Test_Gold.xml')

EN_SEMEVAL_SPLIT_DATA_TRAIN = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_SPLIT_EN, 'Restaurants_Train-TRAIN.json')
EN_SEMEVAL_SPLIT_DATA_TEST = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_SPLIT_EN, 'Restaurants_Test_Gold-TEST.json')
EN_SEMEVAL_SPLIT_DATA_DEV = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_SPLIT_EN, 'Restaurants_Train-DEV.json')

CS_SEMEVAL_2014_RAW_DATASET = os.path.join(RAW_DATASETS_DIR, 'czech', 'CzechABSA-v2_format_SemEval2014.xml')
CS_SEMEVAL_SPLIT_DATA_TRAIN = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_SPLIT_CS, 'CzechABSA-v2_format_SemEval2014-TRAIN.json')
CS_SEMEVAL_SPLIT_DATA_TEST = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_SPLIT_CS, 'CzechABSA-v2_format_SemEval2014-TEST.json')
CS_SEMEVAL_SPLIT_DATA_DEV = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_SPLIT_CS, 'CzechABSA-v2_format_SemEval2014-DEV.json')

SRL_DATA_DIR = os.path.join(GENERATED_DATASET_DIR, 'srl')


SEMEVAL_2014_ASPECT_CATEGORIES = ['price', 'anecdotes/miscellaneous', 'food', 'ambience', 'service']

# Pozor pouzivaji se pro vsechny ulohy
EN_ASPECT_CATEGORIES_QUESTIONS_STRINGS = {'price' : 'price', 'anecdotes/miscellaneous' : 'anecdotes',
                                          'food' : 'food', 'ambience' : 'ambience', 'service' : 'service'}
CS_ASPECT_CATEGORIES_QUESTIONS_STRINGS = {'price' : 'cena', 'anecdotes/miscellaneous' : 'ostatní',
                                          'food' : 'jídlo', 'ambience' : 'prostředí', 'service' : 'obsluha'}

SEMEVAL_2014_EN_LABELS_GENERATING = ['positive', 'neutral', 'negative', 'conflict', 'none']
SEMEVAL_2014_EN_GENERATED_LABELS_BINARY_MAP_INDICES = {0 : 'positive', 1 : 'neutral', 2 : 'negative', 3:'conflict', 4:'none'}

SEMEVAL_2014_CS_LABELS_GENERATING = ['positive', 'neutral', 'negative', 'none']
SEMEVAL_2014_CS_GENERATED_LABELS_BINARY_MAP_INDICES = {0 : 'positive', 1 : 'neutral', 2 : 'negative', 3:'none'}

EN_SEMEVAL_2014_LABELS_GENERATING_MAPING = {'positive': 'positive', 'neutral': 'neutral', 'negative': 'negative',
                                            'conflict': 'conflict', 'none': 'none'}

CS_SEMEVAL_2014_LABELS_GENERATING_MAPING = {'positive': 'pozitivní', 'neutral': 'neutrální', 'negative': 'negativní',
                                            'none': 'žádná'}

SEMEVAL_2014_EN_LABELS_CUSTOM_QA = ['positive', 'neutral', 'negative', 'conflict']
SEMEVAL_2014_CS_LABELS_CUSTOM_QA = ['positive', 'neutral', 'negative']

# Important to reproduce results
RANDOM_SEED = 666

# Size of dev data, proportion that is taken from train data
DEV_SIZE_EN = 0.1
DEV_SIZE_CS = 0.1

# Size of test data for datasets that are not split, proportion that is taken from
TEST_SIZE_CS = 0.2

CS_NUMBER_FOLDS = 10


MODES = ['crosslingual', 'monolingual']
CROSS_LINGUAL_MODE = 'crosslingual'
MONOLINGUAL_MODE = 'monolingual'

ENGLISH = 'english'  # english
CZECH = 'czech'  # czech

