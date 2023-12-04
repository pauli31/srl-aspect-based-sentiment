import argparse
import logging
import os

from config import DATASET_LANGS, DATASET_SOURCES, LOGGING_FORMAT, LOGGING_DATE_FORMAT, LOG_DIR, \
    TRANSFORMERS_TRAINED_MODELS, TASK_TYPES, SCHEDULERS, DATASET_LIST, CAT_SOLUTION_TYPES, MONOLINGUAL_MODE, MODES, \
    OPTIMIZERS_CHOICES, INJECTION_MODES
from utils.general_utils import generate_file_name_transformer, generate_wandb_group

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)

def build_parser_fine_tune():
    parser = argparse.ArgumentParser(allow_abbrev=False, description='Fine-tuning Aspect-based sentiment')

    # Required parameters
    parser.add_argument("--dataset_name",
                        required=True,
                        choices=DATASET_LIST,
                        help="The dataset that will be used, they correspond to names of folders")

    parser.add_argument("--model_name",
                        required=True,
                        type=str,
                        help="Name of model from hugging face or path to directory with the model")

    parser.add_argument("--task",
                        required=True,
                        choices=TASK_TYPES,
                        help="The task that is going to be fine-tuned, CAT - Aspect category recognition and classification,"
                             " TERM - Aspect term recognition and classification")

    parser.add_argument("--injection_mode",
                        required=False,
                        choices=INJECTION_MODES,
                        default=None)

    parser.add_argument("--injection_model_name",
                        type=str,
                        help='TODO')

    parser.add_argument("--use_custom_model",
                        default=False,
                        action='store_true',
                        help="It has to be set for models with custom head")

    # Optional parameters
    parser.add_argument("--solution_type_cat",
                        default='QA_B',
                        choices=CAT_SOLUTION_TYPES,
                        help="The type of solution used for the aspect category task, only applied when --task is set to CAT")

    parser.add_argument("--lang_mode",
                        default=MONOLINGUAL_MODE,
                        choices=MODES,
                        help='Mode that will be used for training, if monolingual only monolingual model is trained and evaluated')

    parser.add_argument("--max_seq_len",
                        default=512,
                        type=int,
                        help="Maximum sequence length of tokens  used as an input for the model")

    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size")

    parser.add_argument("--epoch_num",
                        default=5,
                        type=int,
                        help="Number of epochs for fine tuning")

    parser.add_argument("--lr",
                        default=2e-6,
                        type=float,
                        help="Learning rate")

    parser.add_argument("--classifier_dropout",
                        default=None,
                        type=float,
                        help="Dropout for classifier layer,"
                             " if not set the parameter --hidden_dropout_prob is used instead")

    parser.add_argument("--hidden_dropout_prob",
                        default=0.1,
                        type=float,
                        help="Dropout for hidden layers")

    parser.add_argument("--scheduler",
                        default='linear_wrp',
                        choices=SCHEDULERS,
                        type=str,
                        help="Schedulre used for scheduling learning rate,"
                             " see https://huggingface.co/transformers/main_classes/optimizer_schedules.html#learning-rate-schedules-pytorch")

    parser.add_argument("--warm_up_steps",
                        default=0,
                        type=float,
                        help="Number of warmup steps, if less than 1 than it is used as percents/fraction of the total"
                             " number of steps, cannot be set to one")

    parser.add_argument("--optimizer",
                        default='AdamW',
                        choices=OPTIMIZERS_CHOICES,
                        help="Optimizer one of: " + str(OPTIMIZERS_CHOICES))

    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay parameter, is only applied when AdamW optimizer is used")

    parser.add_argument("--adadelta_rho",
                        default=0.6,
                        type=float,
                        help="Rho value for the adadelta optimizer, is only applied when Adadelta is used")

    # It is set in the args_utils in init_logging
    # parser.add_argument("--current_fold",
    #                     default=0,
    #                     type=int,
    #                     help="The fold that will be used, if cross-validation dataset is used")

    parser.add_argument("--fold_from",
                        default=-1,
                        type=int,
                        help="The fold that will start in cross-validation")

    parser.add_argument("--fold_to",
                        default=-1,
                        type=int,
                        help="The final fold in cross-validation")

    parser.add_argument("--use_cpu",
                        default=False,
                        action='store_true',
                        help="If set, the program will always run on CPU")

    parser.add_argument("--from_tf",
                        default=False,
                        action='store_true',
                        help="If set, the program will try to load the tensorflow model into pytorch model, in that case"
                             " all GPUs for tensorflow are disabled")

    parser.add_argument("--use_automodel",
                        default=False,
                        action='store_true',
                        help="If set, the program will try to load model without specified the --model_type,"
                             " and --tokenizer_type params")


    parser.add_argument("--model_type",
                        default='bert',
                        choices=['bert', 'albert', 'xlm', 'xlm-r', 'mt5', 'robeczech','electra'],
                        help="Type of model that will be loaded")

    parser.add_argument("--tokenizer_type",
                        default='berttokenizer',
                        choices=['berttokenizer', 'berttokenizerfast', 'xlmtokenizer', 'xlm-r-tokenizer',
                                 'berttokenizerfast-cased',
                                 'mt5-tokenizer', 'robeczech-tokenizer', 'electra-fast'],
                        help="Type of tokenizer that will be used, the tokenizer config must be in"
                             " the same folder as the model, specified by parameter model_name")

    parser.add_argument("--use_random_seed",
                        default=False,
                        action='store_true',
                        help="If set, the program will NOT set a seed value for all random sources, "
                             "if set the results should NOT be same across runs with the same configuration.")

    # Watch the cross-lingual datasets
    parser.add_argument("--use_only_train_data",
                        default=False,
                        action='store_true',
                        help="If set, the program will use training and develoopment data for training, i.e. it will use"
                             "train + dev for training, no validation is done during training")

    parser.add_argument("--max_train_data",
                        default=-1,
                        type=float,
                        help="Amount of data that will be used for training, "
                             "if (-inf, 0> than it is ignored, "
                             "if (0,1> than percentage of the value is used as training data, "
                             "if (1, inf) absolute number of training examples is used as training data")

    parser.add_argument("--data_loader_num_workers",
                        default=4,
                        type=int,
                        help="Number of workers for data loading, the value is passed into DataLoader object, "
                             "for debug set to 1")

    parser.add_argument("--enable_wandb",
                        default=False,
                        action='store_true',
                        help="If set, the program will use wandb for logging, otherwise not")

    parser.add_argument("--enable_masking",
                        default=False,
                        action='store_true',
                        help="If set, the program will mask the pad tokens in average injection mode only")

    parser.add_argument("--data_parallel",
                        default=False,
                        action='store_true',
                        help='If set, the program will run on all avaialble GPUs')

    parser.add_argument("--num_repeat",
                        default=1,
                        type=int,
                        help="Specify the number, of how many times will be the experiment repeated")


    parser.add_argument("--freze_base_model",
                        default=False,
                        action='store_true',
                        help="If set, the program will freeze all parametrs from base model (except the last linear layer) "
                             "It does NOT WORK for custom model, must be implemented")

    parser.add_argument("--save_model",
                        default=False,
                        action='store_true',
                        help='If set, the program will save the trained model')

    parser.add_argument("--model_save_dir",
                        default=TRANSFORMERS_TRAINED_MODELS,
                        type=str,
                        help="Folder where the finetuned model will be saved, default is trained_models/transformers")

    parser.add_argument("--print_stat_frequency",
                        default=25,
                        type=int,
                        help="Specify the frequency of printing train info, i.e. after how many batches will be the "
                             "info printed")

    parser.add_argument("--draw_dataset_stats",
                        action="store_true",
                        help="If specified the statistics about the given datasets are printed and saved to dataset folder"
                             "don't forget to specify the correct tokenizer, it can be slower because it loads the entire dataset"
                             " and it tokenizes it, The fine-tuning is not run with this parameter")

    parser.add_argument("--silent",
                        default=False,
                        action='store_true',
                        help="If used, logging is set to ERROR level, otherwise INFO is used")

    parser.add_argument("--srl_other_weight",
                        default=1,
                        type=float,
                        help="To decrease the weight for OTHER class in cross entropy loss of SRL")

    parser.add_argument("--disable_output_bias",
                        default=False,
                        action='store_true',
                        help="If set, the model will not use bias on the output")

    parser.add_argument("--simulate_gold_preds",
                        default=False,
                        action='store_true',
                        help="If set, only arguments for the gold predicates will bw predicted")

    parser.add_argument("--end2end",
                        default=False,
                        action='store_true',
                        help="If set, end2end model will be used")

    parser.add_argument("--srl_use_dot_product",
                        default=False,
                        action='store_true',
                        help="If set, SRL model with use dot product on the top of the encoder")

    parser.add_argument("--srl_separate_encoding",
                        default=False,
                        action='store_true',
                        help="If set, separate representation for predicates and arguments will be used")

    parser.add_argument("--srl_official_eval",
                        default=False,
                        action='store_true',
                        help="If set, official scoring script will be used for test evaluation")

    parser.add_argument("--use_pre_trained_srl_model",
                        default=False,
                        action='store_true',
                        help='If true, pre-trained mode lis used for Custom ABSA models')

    parser.add_argument('--dataset_lang',
                        default="cs",
                        choices=DATASET_LANGS,
                        help="Language of the dataset")


    parser.add_argument("--pre_trained_srl_model_path",
                        help='Path to pre trained SRL model ')

    parser.add_argument("--predicate_sampling_prob",
                        default=1.0,
                        type=float,
                        help="Negative sampling probability for predicate identification. Makes sense only if --simulate_gold_preds is False")

    return parser




def build_parser_generating():
    parser = argparse.ArgumentParser(allow_abbrev=False, description='Data generation utils')

    parser.add_argument("--split_data",
                        default=False,
                        action="store_true",
                        help="Whether or not to split the data")

    parser.add_argument('--dataset_lang',
                        required=True,
                        choices=DATASET_LANGS,
                        help="Language of the dataset")

    parser.add_argument('--source',
                        required=True,
                        choices=DATASET_SOURCES,
                        help="Sources of datasets")

    return parser


# needed for wandb grouping
# https://docs.wandb.ai/guides/track/advanced/grouping
def set_general_group_config(args, parser,generating_fce=generate_wandb_group):
    grouping_config = generating_fce(args)
    grouping_config = grouping_config + "-group"
    parser.add_argument("--wandb_custom_group",
                        default=grouping_config)

    args = parser.parse_args()

    return args


def init_logging(args, parser, result_file, current_fold=-1, generating_fce=generate_file_name_transformer, set_format=True):

    config_name = generating_fce(args)

    file_name = os.path.join(LOG_DIR, config_name + '.log')
    parser.add_argument("--config_name",
                        default=config_name)
    parser.add_argument("--result_file",
                        default=result_file)

    parser.add_argument("--current_fold",
                        default=current_fold)
    args = parser.parse_args()

    if set_format:
        # just to reset logging settings
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(format=LOGGING_FORMAT,
                            datefmt=LOGGING_DATE_FORMAT,
                            filename=file_name)

        formatter = logging.Formatter(fmt=LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logging.root.setLevel(level=logging.INFO)
        if args.silent is True:
            # logging.root.setLevel(level=logging.ERROR)
            console_handler.setLevel(level=logging.ERROR)
        else:
            # logging.root.setLevel(level=logging.INFO)
            console_handler.setLevel(level=logging.INFO)

        logging.getLogger().addHandler(console_handler)

    return args