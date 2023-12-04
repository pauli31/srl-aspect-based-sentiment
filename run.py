import logging
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from args_utils import build_parser_fine_tune, init_logging, set_general_group_config
import sys






from config import RESULTS_DIR, LOGGING_FORMAT, LOGGING_DATE_FORMAT, WANDB_API_KEY_DIR
from fine_tuning.fine_tuning_torch import fine_tune_torch
from utils.general_utils import generate_file_name_transformer


logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)

# https://github.com/LorenzoAgnolucci/BERT_for_ABSA/blob/master/BERT_for_ABSA.ipynb
# https://github.com/ThilinaRajapakse/simpletransformers/issues/515
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(args):

    if args.task == 'CAT' or args.task == 'TERM' or args.task == 'SRL':
        fine_tune_torch(args)
    elif args.task == 'PRE-TRAIN':

        raise NotImplementedError("Pre-training not implemented yet, implement it!")
    else:
        raise Exception("Unknown type of task")
    pass


def load_wandb_api_key(path):
    with open(path, "r", encoding='utf-8') as f:
        data = f.read().replace('\n', '')

    data = data.strip()
    return data


def main():
    print('Hello, Aspect-Based Sentiment Experiments Fine-tuning')
    if RUN_LOCAL is False:
        wandb_api_key = load_wandb_api_key(WANDB_API_KEY_DIR)
        os.environ["WANDB_API_KEY"] = wandb_api_key
        os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"


    parser = build_parser_fine_tune()
    args = parser.parse_args()
    result_file = generate_file_name_transformer(args)

    result_file = result_file + ".results"
    result_file = os.path.join(RESULTS_DIR, result_file)


    if args.fold_from == -1:
        # because we have to do one iteration even if there are no folds
        num_folds = 1
        folds_start = 0
        folds_end = 1
    else:
        folds_start = args.fold_from
        # + 1 bcs of range
        folds_end = args.fold_to + 1

    args = set_general_group_config(args, parser)
    os.environ["WANDB_RUN_GROUP"] = args.wandb_custom_group

    num_repeat = args.num_repeat + 1
    for repeat in range(1, num_repeat):
        logger.info("Running repeat:" + str(repeat))

        for current_fold in range(folds_start, folds_end):
            logger.info("Running fold:" + str(current_fold))
            parser = build_parser_fine_tune()
            args = parser.parse_args()
            args = init_logging(args, parser, result_file, current_fold=current_fold)

            logger.info(f"Running fine-tuning with the following parameters:{args}")
            logger.info("-------------------------")

            run(args)

            logger.info("Fold completed:" + str(current_fold))
            logger.info(70 * "#")
            logger.info(70 * "#")
            logger.info(70 * "#")

        logger.info("Run completed")
        logger.info("----------------------------------------------------")




RUN_LOCAL = False

def set_local_settings():
    if RUN_LOCAL is True:

        # sys.argv.extend(['--dataset_name', 'semeval2014_en'])
        # sys.argv.extend(['--dataset_name', 'semeval2014_cs'])
        # sys.argv.extend(['--dataset_name', 'semeval2014_cs-cross_val'])

        # sys.argv.extend(['--dataset_name', 'cs_absa_srl_dataset'])
        sys.argv.extend(['--dataset_name', 'cs_srl_e2e'])
        # sys.argv.extend(['--use_only_train_data'])
        sys.argv.extend(['--epoch_num', '3'])
        # sys.argv.extend(['--fold_from', '0'])
        # sys.argv.extend(['--fold_to', '9'])

        # sys.argv.extend(['--model_name', 'bert-base-cased'])


        # sys.argv.extend(['--model_name', 'bert-large-cased'])
        # sys.argv.extend(['--task', 'CAT'])

        # jenom pro SrlBert
        # sys.argv.extend(['--task', 'SRL'])

        sys.argv.extend(['--epoch_num', '5'])
        # sys.argv.extend(['--max_seq_len', '250'])
        sys.argv.extend(['--max_seq_len', '20'])
        sys.argv.extend(['--solution_type_cat', 'NLI_M'])
        # sys.argv.extend(['--solution_type_cat', 'NLI_B'])
        sys.argv.extend(['--data_loader_num_workers', '0'])
        # sys.argv.extend(['--enable_wandb'])
        # sys.argv.extend(['--save_model'])
        # sys.argv.extend(['--num_repeat', '6'])


        sys.argv.extend(['--batch_size', '32'])
        sys.argv.extend(['--dataset_lang', 'en'])
        # sys.argv.extend(['--hidden_dropout_prob', '0.2'])
        # sys.argv.extend(['--classifier_dropout', '0.3'])

        # sys.argv.extend(['--tokenizer_type', 'berttokenizerfast-cased'])
        # sys.argv.extend(['--model_name', './data/local_models/czert-bert-base-cased'])
        # sys.argv.extend(['--model_name', './data/local_models/Czert-restaurants'])
        # sys.argv.extend(['--model_name', './data/local_models/Czert-reviews'])
        # sys.argv.extend(['--from_tf'])
        sys.argv.extend(['--print_stat_frequency','25'])
        # sys.argv.extend(['--draw_dataset_stats'])

        sys.argv.extend(['--use_automodel'])
        # sys.argv.extend(['--model_name', 'ufal/robeczech-base'])
        # sys.argv.extend(['--model_name', 'xlm-roberta-large'])

        # sys.argv.extend(['--model_name', 'Seznam/small-e-czech'])
        sys.argv.extend(['--model_name', 'google/electra-small-discriminator'])
        sys.argv.extend(['--tokenizer_type', 'electra-fast'])
        sys.argv.extend(['--model_type', 'electra'])
        # sys.argv.extend(['--injection_mode', 'concat-avg'])
        # sys.argv.extend(['--use_custom_model'])


        # sys.argv.extend(['--injection_mode', 'multi-task'])
        # sys.argv.extend(['--injection_mode', 'multi-task'])
        # sys.argv.extend(['--injection_model_name', 'Seznam/small-e-czech'])
        # sys.argv.extend(['--injection_model_name', './data/local_models/czert-bert-base-cased'])
        # sys.argv.extend(['--model_name', './data/local_models/czert-bert-base-cased'])
        # sys.argv.extend(['--from_tf'])
        # sys.argv.extend(['--use_cpu'])

        # e2e
        sys.argv.extend(['--end2end'])
        sys.argv.extend(['--task', 'SRL'])
        sys.argv.extend(['--use_custom_model'])


        # concat-avg
        # sys.argv.extend(['--injection_mode', 'concat-avg'])
        # sys.argv.extend(['--use_custom_model'])
        # sys.argv.extend(['--task', 'CAT'])
        # sys.argv.extend(['--dataset_name', 'semeval2014_cs'])
        # sys.argv.extend(['--dataset_name', 'semeval2014_en'])
        # sys.argv.extend(['--solution_type_cat', 'NLI_M'])
        # sys.argv.extend(['--enable_masking'])

        # sys.argv.extend(['--use_pre_trained_srl_model'])
        # sys.argv.extend(['--pre_trained_srl_model_path','./data/local_models/Czert-SRL-Pre-trained/model'])
        # sys.argv.extend(['--pre_trained_srl_model_path','./trained_models/transformers/small-e-czech/e2e-srl-test'])

        # SRLAvergage - average
        # sys.argv.extend(['--injection_mode', 'average'])
        # sys.argv.extend(['--task', 'CAT'])
        # sys.argv.extend(['--dataset_name', 'semeval2014_cs'])
        # sys.argv.extend(['--solution_type_cat', 'NLI_M'])
        #
        # sys.argv.extend(['--use_pre_trained_srl_model'])
        # sys.argv.extend(['--pre_trained_srl_model_path', './data/local_models/Czert-SRL-Pre-trained/model'])
        #
        # # Multitask
        # sys.argv.extend(['--injection_mode', 'multi-task'])
        # sys.argv.extend(['--dataset_name', 'cs_absa_srl_dataset'])
        # sys.argv.extend(['--dataset_name', 'en_absa_srl_dataset'])
        # sys.argv.extend(['--use_custom_model'])





if __name__ == '__main__':
    set_local_settings()
    main()

