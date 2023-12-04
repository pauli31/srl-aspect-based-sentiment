import logging
import random
import numpy as np
import torch
import gc
import os
from collections import Counter
import seaborn as sns
import wandb
import time

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertTokenizerFast, XLMTokenizer, XLMRobertaTokenizer, T5Tokenizer, \
    AutoTokenizer, BertForSequenceClassification, AlbertForSequenceClassification, XLMForSequenceClassification, \
    XLMRobertaForSequenceClassification, AutoConfig, AutoModelForSequenceClassification, \
    ElectraForSequenceClassification, ElectraTokenizerFast, AutoModel, AutoModelForTokenClassification

from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT, RANDOM_SEED, CROSS_LINGUAL_MODE, WANDB_DIR, \
    INJECTION_MODE_AVERAGE, INJECTION_MODE_CONCAT_AVERAGE, INJECTION_MODE_CONCAT_CONVOLUTION, INJECTION_MODE_MULTI_TASK
from data_load.dataset import build_data_loader
from data_load.loader import DATASET_LOADERS
from fine_tuning.evaluation import evaluate_predictions, compute_aspect_cat_semeval_PRF, prepare_labels4aspect_cat_eval, \
    compute_semeval_acc, GeneralResult, AspectEvalResult, CategoryEvaluationResult, print_category_results
from fine_tuning.models import SRLAverageModel, SRLConcatModel, MultiTaskModel, SrlBert, SrlBertSimple
from fine_tuning.train import run_training, get_predictions
from utils.general_utils import format_time, save_model_transformer, load_raw_model
from utils.nn_utils import get_optimizer, get_lr_scheduler
import utils.utils_srl

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)


class TorchTuner(object):
    def __init__(self, args):
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch_num
        self.dataset_loader = DATASET_LOADERS[args.dataset_name](args)
        self.use_only_train_data = args.use_only_train_data
        self.data_parallel = args.data_parallel

        if args.use_random_seed is False:
            # init RANDOM_SEED
            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            torch.manual_seed(RANDOM_SEED)
            torch.cuda.manual_seed_all(RANDOM_SEED)
        if args.use_cpu is True:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Running fine-tuning on:{self.device}")
        print_gpu_info()

        try:
            GPUs_count = torch.cuda.device_count()
            logger.info("We have available more GPUs:" + str(GPUs_count))

            logger.info("We try to run it on multiple GPUs:" + str(self.data_parallel))

        except Exception as e:
            logger.info("Trying to init data parallelism")

        self.tokenizer = self.load_tokenizer(args)

        # Load dataset
        logger.info("Loading datasets...")
        self.print_dataset_info(args)
        if self.use_only_train_data:
            self.train_data_loader = build_data_loader(self.dataset_loader.get_train_dev_data(), self.tokenizer, args,
                                                       True)
            self.dev_data_loader = None

            self.train_size = len(self.dataset_loader.get_train_dev_data())
            self.dev_size = 0
        else:
            if args.epoch_num > 0:
                self.train_data_loader = build_data_loader(self.dataset_loader.get_train_data(), self.tokenizer, args, True)
                self.train_size = len(self.dataset_loader.get_train_data())
            else:
                self.train_size = 0
                self.train_data_loader = build_data_loader([], self.tokenizer, args, False)

            self.dev_data_loader = build_data_loader(self.dataset_loader.get_dev_data(), self.tokenizer, args)
            self.dev_size = len(self.dataset_loader.get_dev_data())

        self.test_size = len(self.dataset_loader.get_test_data())
        self.num_labels = self.dataset_loader.get_class_num()

        if args.injection_mode == INJECTION_MODE_MULTI_TASK:
            self.srl_num_labels = self.dataset_loader.get_class_num_srl()

        self.test_data_loader = build_data_loader(self.dataset_loader.get_test_data(), self.tokenizer, args)

        logger.info("Train size: " + str(self.train_size))
        logger.info("Dev size: " + str(self.dev_size))
        logger.info("Test size: " + str(self.test_size))

        if self.dataset_loader.get_lang_mode() == CROSS_LINGUAL_MODE:
            self.dev_target_size = len(self.dataset_loader.get_target_lang_dev_data())
            self.dev_target_data_loader = build_data_loader(self.dataset_loader.get_target_lang_dev_data(),
                                                            self.tokenizer, args)

            logger.info("Size of dev data for target language: " + str(self.dataset_loader.get_target_lang()) + " is:" +
                        str(self.dev_target_size))
            logger.info("Source language: " + str(self.dataset_loader.get_source_lang()))
            logger.info("Target language: " + str(self.dataset_loader.get_target_lang()))
        else:
            self.dev_target_size = 0
            self.dev_target_data_loader = None

        logger.info("Datasets loaded")
        logger.info("Running in language mode:" + str(args.lang_mode))
        logger.info(f"Number of labels in dataset:{self.num_labels}")

    def set_dropouts(self, model, args):
        hidden_dropout = args.hidden_dropout_prob
        classifier_dropout = args.classifier_dropout

        if classifier_dropout is None:
            classifier_dropout = hidden_dropout
            args.classifier_dropout = hidden_dropout

        if args.injection_mode == INJECTION_MODE_AVERAGE:
            model.set_dropout(classifier_dropout, hidden_dropout)
        else:
            model.config.classifier_dropout = classifier_dropout
            model.config.hidden_dropout_prob = hidden_dropout



    def set_dropouts_args(self, args):
        hidden_dropout = args.hidden_dropout_prob
        classifier_dropout = args.classifier_dropout

        if classifier_dropout is None:
            args.classifier_dropout = hidden_dropout

    def fine_tune_cat_task(self, args):

        self.set_dropouts_args(args)

        # load model
        logger.info("Loading model...")
        model = self.load_model(args)
        # set the dropouts
        self.set_dropouts(model, args)
        logger.info("Model loaded")

        if args.data_parallel is True:
            if torch.cuda.device_count() > 1:
                logger.info("Trying to apply data parallesism, number of used GPUs:" + str(torch.cuda.device_count()))
                # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
                model = torch.nn.DataParallel(model)
            else:
                logger.info("Data parallelism is enabled but there is only GPUs:" + str(torch.cuda.device_count()))

        if args.enable_wandb is True:
            wandb_tmp_name = str(args.config_name)
            try:
                wandb.init(project="srl-aspect-based-sentiment-final", entity='srl-aspect-based',
                           config=vars(args),
                           dir=WANDB_DIR,
                           reinit=True)
            except Exception as e:
                logger.error("Error WANDB with exception e:" + str(e))
            # If we run multiple runs in one python script, there is a memory leak in wandb causing OOM on GPU after
            # some time
            # wandb.watch(model)

        # move it to device
        model = model.to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
        logger.info("Model moved to device")

        # Init optimizer
        optimizer = get_optimizer(model.parameters(), args)
        total_steps = len(self.train_data_loader) * self.epoch_num
        scheduler = get_lr_scheduler(args, optimizer, total_steps)

        logger.info('Training model on train data...')
        t0 = time.time()

        history = run_training(args, model, self.device, loss_fn, optimizer, scheduler, self,
                               self.train_data_loader, self.dev_data_loader, self.dev_target_data_loader)

        train_time = time.time() - t0
        logger.info(f'Total time for training:{format_time(train_time)}')

        print(70 * "$$$")
        print(70 * "$$$")
        logger.info("Training done")

        self.perform_train_eval(model, args)

        if args.save_model:
            save_custom = args.use_custom_model
            save_model_transformer(model, self.tokenizer, optimizer, args, args.model_save_dir, save_custom)

        return history

    def fine_tune_term_task(self, args):
        raise NotImplementedError("Not implemented yet, do it now!")

    # in order to preserve the v3.x behavior we have to set
    # https://huggingface.co/transformers/migration.html
    # return dict as true
    # https://huggingface.co/transformers/pretrained_models.html
    def load_model(self, args):
        if args.injection_mode != None:
            model_absa = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=self.num_labels,
                                                                            from_tf=args.from_tf,)
            # model_srl = AutoModel.from_pretrained('UWB-AIR/Czert-B-base-cased')

            if args.injection_mode == INJECTION_MODE_AVERAGE:
                if args.use_pre_trained_srl_model is True:
                    model_path = os.path.join(args.pre_trained_srl_model_path, 'model_torch.bin')
                    pre_trained_model, _ = load_raw_model(model_path)
                    model_srl = pre_trained_model.base_model
                else:
                    model_srl = AutoModel.from_pretrained(args.injection_model_name)
                model = SRLAverageModel(model_absa, model_srl)
            elif args.injection_mode == INJECTION_MODE_CONCAT_AVERAGE or args.injection_mode == INJECTION_MODE_CONCAT_CONVOLUTION:
                if args.use_pre_trained_srl_model is True:
                    model_path = os.path.join(args.pre_trained_srl_model_path, 'model_torch.bin')
                    pre_trained_model, _ = load_raw_model(model_path)
                    model_srl = pre_trained_model.base_model
                else:
                    model_srl = AutoModel.from_pretrained(args.injection_model_name)
                model = SRLConcatModel(model_absa, model_srl, args, self.num_labels)
            elif args.injection_mode == INJECTION_MODE_MULTI_TASK:
                model = MultiTaskModel(model_absa, self.num_labels, self.srl_num_labels, args)
            else:
                raise Exception("Unknown injection model:" + str(args.injection_mode))
        elif args.task == "SRL":
            if args.end2end:
                model = SrlBert(AutoModel.from_pretrained(args.model_name,
                                                          num_labels=self.num_labels,
                                                          from_tf=args.from_tf), self.num_labels, args)
            else:
                model = SrlBertSimple(AutoModelForTokenClassification.from_pretrained(args.model_name,
                                                                        num_labels=self.num_labels,
                                                                        from_tf=args.from_tf), args)
        else:
            if args.use_automodel is True:
                model = AutoModelForSequenceClassification.from_pretrained(args.model_name,
                                                                           num_labels=self.num_labels,
                                                                           from_tf=args.from_tf,
                                                                           return_dict=False)
            else:
                if args.model_type == 'bert':
                    model = BertForSequenceClassification.from_pretrained(args.model_name,
                                                                          num_labels=self.num_labels,
                                                                          from_tf=args.from_tf,
                                                                          return_dict=False)
                elif args.model_type == 'albert':
                    model = AlbertForSequenceClassification.from_pretrained(args.model_name,
                                                                            num_labels=self.num_labels,
                                                                            from_tf=args.from_tf,
                                                                            return_dict=False)
                elif args.model_type == 'xlm':
                    model = XLMForSequenceClassification.from_pretrained(args.model_name,
                                                                         num_labels=self.num_labels,
                                                                         from_tf=args.from_tf,
                                                                         return_dict=False)
                elif args.model_type == 'xlm-r':
                    model = XLMRobertaForSequenceClassification.from_pretrained(args.model_name,
                                                                                num_labels=self.num_labels,
                                                                                from_tf=args.from_tf,
                                                                                return_dict=False)
                elif args.model_type == 'electra':
                    model = ElectraForSequenceClassification.from_pretrained(args.model_name,
                                                                             num_labels=self.num_labels,
                                                                             from_tf=args.from_tf,
                                                                             return_dict=False)
                elif args.model_type == 'mt5':
                    config = AutoConfig.from_pretrained('t5-small')
                    model = AutoModelForSequenceClassification.from_config(config)
                    print("Does not work")

                elif args.model_type == 'robeczech':
                    # config = AutoConfig.from_pretrained(args.bert)
                    # config.output_hidden_states = True
                    model = AutoModelForSequenceClassification.from_pretrained(args.model_name,
                                                                               num_labels=self.num_labels,
                                                                               return_dict=False)
                else:
                    raise Exception(f"Unkown model type:{args.model_type}")

        if args.freze_base_model is True:
            logger.info("Freezing base model layers")
            for name, param in model.base_model.named_parameters():
                print(name)
                param.requires_grad = False

        try:
            # pp = 0
            # for p in list(model.parameters()):
            #     nn = 1
            #     for s in list(p.size()):
            #         nn = nn * s
            #     pp += nn
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            pp = sum([np.prod(p.size()) for p in model_parameters])
            logger.info("Number of parameters for model:" + str(args.model_name) + " is:" + str(pp))
        except Exception as e:
            logger.error("Error during count number:" + str(e))

        return model

    def perform_train_eval(self, model, args):
        logger.info("Evaluating model")
        # Test data
        logger.info("Getting predictions for test data...")
        texts_orig, texts_help, y_pred, y_pred_probs, y_true, _, _, \
        y_pred_srl, y_pred_probs_srl, y_true_srl = get_predictions(model, self.test_data_loader,
                                                                   self.device, args.batch_size, args, None)
        if len(y_pred) > 0:
            logger.info("Got predictions for test data")
            logger.info("Performing evaluation for test data")
            test_result = self.eval_model(y_true, y_pred, y_pred_probs)
            logger.info("Evaluation done for test data")
            results_head, results = print_category_results("test", test_result, args.enable_wandb, args)

            result_string = "\n-----------Test Results------------\n" + results_head + "\n" + results
            only_results = results + '\t'

        if len(y_pred_srl) > 0:
            logger.info("Got predictions for test data")
            logger.info("Performing evaluation for test data")
            test_result = utils.utils_srl.eval_srl(y_true_srl, y_pred_srl, y_pred_probs_srl, lang=args.dataset_lang, end2end=args.end2end, simulate_gold_preds=args.simulate_gold_preds)
            logger.info("Evaluation done for test data")
            results_head, results = utils.utils_srl.print_eval_srl(test_result, args.enable_wandb, "SRL-test")

            result_string = "\n-----------Test Results------------\n" + results_head + "\n" + results
            only_results = results + '\t'
            if args.srl_official_eval:
                test_result_official = utils.utils_srl.eval_srl(y_true_srl, y_pred_srl, y_pred_probs_srl, lang=args.dataset_lang, official=True, end2end=args.end2end, simulate_gold_preds=args.simulate_gold_preds)
                utils.utils_srl.print_eval_srl(test_result_official, args.enable_wandb, "SRL-official-test")

        # DEV data
        if args.use_only_train_data is False:
            logger.info("Getting predictions for dev data...")
            texts_orig_dev, texts_help_dev, y_pred, y_pred_probs, y_true, _, _, \
            y_pred_srl, y_pred_probs_srl, y_true_srl = get_predictions(model, self.dev_data_loader,
                                                                       self.device, args.batch_size, args, None)
            logger.info("Got predictions for DEV data")
            logger.info("Performing evaluation for DEV data")

            if len(y_pred_srl) > 0:
                dev_result = utils.utils_srl.eval_srl(y_true_srl, y_pred_srl, y_pred_probs_srl, lang=args.dataset_lang, end2end=args.end2end)
                logger.info("Evaluation done for DEV data")
                dev_results_head, dev_results = utils.utils_srl.print_eval_srl(dev_result, args.enable_wandb, "SRL-dev")

                result_string += "\n-----------Dev Results------------\n" + dev_results_head + "\n" + dev_results
                only_results += dev_results

            if len(y_pred) > 0:
                dev_result = self.eval_model(y_true, y_pred, y_pred_probs)
                logger.info("Evaluation done for DEV data")
                dev_results_head, dev_results = print_category_results("dev", dev_result, args.enable_wandb, args)

                result_string += "\n-----------Dev Results------------\n" + dev_results_head + "\n" + dev_results
                only_results += dev_results


        dataset_name = args.dataset_name
        config_str = f'{dataset_name}\tTransformer train test:{args.model_name} {args}'

        print("\n\n\n-----------RESULTS results------------\n" + str(result_string) + "\n\n\n")

        only_results = config_str + only_results
        print("\n\n\n-----------Save results------------\n" + str(only_results) + "\n\n\n")

        results_file = args.result_file
        with open(results_file, "a") as f:
            f.write(only_results + "\n")

    def eval_model(self, y_test, y_pred, y_pred_probs):
        # Now we evaluate it on all examples, even the None predictions, that were artificially added
        class_report_full = classification_report(y_test, y_pred, labels=self.dataset_loader.get_classes(),
                                                  target_names=self.dataset_loader.get_class_names())
        print(class_report_full)

        f1_full, accuracy_full, precision_full, recall_full = evaluate_predictions(y_pred, y_test)
        full_res = GeneralResult(f1_full, accuracy_full, precision_full, recall_full, class_report_full)

        result = None

        task = self.args.task
        if task == 'CAT':
            # we have to convert it into the NLI_M or QA_M format
            y_true_aspect, y_pred_aspect, y_pred_probs_aspect = prepare_labels4aspect_cat_eval(y_pred, y_test,
                                                                                               y_pred_probs,
                                                                                               self.args.solution_type_cat,
                                                                                               self.dataset_loader)

            f1mic_aspect, prec_aspect, rec_aspect = compute_aspect_cat_semeval_PRF(y_true_aspect, y_pred_aspect,
                                                                                   self.dataset_loader.get_classes(
                                                                                       eval=True),
                                                                                   [
                                                                                       self.dataset_loader.get_label4text_evaluation(
                                                                                           'none')],
                                                                                   self.dataset_loader.get_aspect_classes_num())

            aspect_res = AspectEvalResult(f1mic_aspect, prec_aspect, rec_aspect)

            # 4 Sentiment classes
            none_class = self.dataset_loader.get_label4text_evaluation('none')
            all_classes = self.dataset_loader.get_classes(eval=True)
            all_classes.remove(none_class)
            all_classes_names = [self.dataset_loader.get_text4label_evaluation(x) for x in all_classes]
            f1_macro_four, acc_macro_four, prec_macro_four, rec_macro_four, class_report_four = \
                compute_semeval_acc(y_true_aspect, y_pred_aspect, y_pred_probs_aspect, all_classes, none_class,
                                    all_classes_names)

            four_res = GeneralResult(f1_macro_four, acc_macro_four, prec_macro_four, rec_macro_four, class_report_four)

            # 3 Sentiment classes
            # for Czech it will be the same as all clases
            # now compute accuracy only for positive, negative and neutral
            try:
                conflict_class = self.dataset_loader.get_label4text_evaluation('conflict')
                all_classes.remove(conflict_class)
                all_classes_names = [self.dataset_loader.get_text4label_evaluation(x) for x in all_classes]
            except Exception as e:
                logger.info("Cannot get class for conflict")

            f1_macro_three, acc_macro_three, prec_macro_three, rec_macro_three, class_report_three = \
                compute_semeval_acc(y_true_aspect, y_pred_aspect, y_pred_probs_aspect, all_classes, none_class,
                                    all_classes_names)

            three_res = GeneralResult(f1_macro_three, acc_macro_three, prec_macro_three, rec_macro_three,
                                      class_report_three)

            # 2 Sentiment classes
            # now remove the neutral class and compute it only for positive and negative
            neutral_class = self.dataset_loader.get_label4text_evaluation('neutral')
            all_classes.remove(neutral_class)
            all_classes_names = [self.dataset_loader.get_text4label_evaluation(x) for x in all_classes]
            f1_macro_binary, acc_macro_binary, prec_macro_binary, rec_macro_binary, class_report_binary = \
                compute_semeval_acc(y_true_aspect, y_pred_aspect, y_pred_probs_aspect, all_classes, none_class,
                                    all_classes_names)

            binary_res = GeneralResult(f1_macro_binary, acc_macro_binary, prec_macro_binary, rec_macro_binary,
                                       class_report_binary)

            # get all the results together
            result = CategoryEvaluationResult(full_res, aspect_res, four_res, three_res, binary_res)
        else:
            raise Exception("Unsupported type of task:" + str(task))

        return result

    @staticmethod
    def load_tokenizer(args):
        if args.use_automodel is True:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        else:
            if args.tokenizer_type == 'berttokenizer':
                tokenizer = BertTokenizer.from_pretrained(args.model_name)

            elif args.tokenizer_type == 'berttokenizerfast':
                tokenizer_path = os.path.abspath(os.path.join(args.model_name, "vocab.txt"))
                print("Tokenizer Path:" + tokenizer_path)
                tokenizer = BertTokenizerFast(tokenizer_path, strip_accents=False)

            elif args.tokenizer_type == 'berttokenizerfast-cased':
                tokenizer_path = os.path.abspath(os.path.join(args.model_name, "vocab.txt"))
                print("Tokenizer Path:" + tokenizer_path)
                tokenizer = BertTokenizerFast(tokenizer_path, strip_accents=False, do_lower_case=False)
                # tokenizer = AutoTokenizer(tokenizer_path, strip_accents=False, do_lower_case=False)

            elif args.tokenizer_type == 'electra-fast':
                tokenizer = ElectraTokenizerFast.from_pretrained(args.model_name)
            elif args.tokenizer_type == 'xlmtokenizer':
                tokenizer = XLMTokenizer.from_pretrained(args.model_name)

            elif args.tokenizer_type == 'xlm-r-tokenizer':
                tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)
            elif args.tokenizer_type == 'mt5-tokenizer':
                tokenizer = T5Tokenizer.from_pretrained(args.model_name)
            elif args.tokenizer_type == 'robeczech-tokenizer':
                tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            else:
                raise Exception(f"Unknown type of tokenizer{args.tokenizer_type}")

        return tokenizer

    def print_dataset_info(self, args):

        if args.injection_mode == INJECTION_MODE_MULTI_TASK:
            dataset_df = self.dataset_loader.absa_loader.load_entire_dataset()
        else:
            if args.task == 'SRL':
                pass
                return
            else:
                dataset_df = self.dataset_loader.load_entire_dataset()

        # just print some example
        help_sentence = dataset_df['help_sentence'][150]
        orig_sentence = dataset_df['orig_sentence'][150]
        id_help = dataset_df['id'][150]
        id_orig = dataset_df['id'][150]
        ids_help = self.tokenizer.encode(help_sentence, max_length=args.max_seq_len, truncation=True,
                                         padding='max_length')
        ids_orig = self.tokenizer.encode(orig_sentence, max_length=args.max_seq_len, truncation=True,
                                         padding='max_length')

        ids_batch_enc = self.tokenizer(help_sentence, orig_sentence,
                                       max_length=args.max_seq_len,
                                       add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                       return_token_type_ids=False,
                                       return_attention_mask=True,
                                       padding='max_length',
                                       truncation=True)

        logger.info(f' Help sentence: {help_sentence}')
        logger.info(f' Help Tokens: {self.tokenizer.convert_ids_to_tokens(ids_help)}')
        logger.info(f' Help Token IDs: {ids_help}')
        logger.info(f' Help Id: {str(id_help)}')
        logger.info("---")
        logger.info(f' Orig sentence: {orig_sentence}')
        logger.info(f' Orig Tokens: {self.tokenizer.convert_ids_to_tokens(ids_orig)}')
        logger.info(f' Orig Token IDs: {ids_orig}')
        logger.info(f' Orig Id: {str(id_orig)}')
        logger.info("---")
        logger.info("Pair...")
        logger.info(f' Help sentence: {help_sentence}')
        logger.info(f' Orig sentence: {orig_sentence}')
        logger.info(f' Pair Tokens: {self.tokenizer.convert_ids_to_tokens(ids_batch_enc["input_ids"])}')
        logger.info(f' Pair Token IDs: {ids_batch_enc["input_ids"]}')

        if self.args.draw_dataset_stats is True:
            logger.info(f"Saving dataset tokens histogram for tokenizer:{self.args.tokenizer_type}")
            # See distribution of text len
            token_lens = []

            count_i = 0
            for txt in dataset_df.orig_sentence:
                tokens = self.tokenizer.encode(txt)
                token_lens.append(len(tokens))
                count_i = count_i + 1
                if count_i % 1000 == 0 and count_i > 0:
                    logger.info("Processed:" + str(count_i))

            max_len = max(token_lens)
            avg_len = np.mean(token_lens)
            cnt = Counter(token_lens)
            # sort by key
            cnt = sorted(cnt.items())
            print("Sentence len - Counts")

            logger.info(f"Max tokens len:{max_len}")
            logger.info(f"Avg tokens len:{avg_len}")

            dataset_name = args.dataset_name
            if dataset_name == 'combined':
                tmp = '-'.join(args.combined_datasets)
                dataset_name = dataset_name + '-' + tmp

            model_name = args.model_name
            model_name = model_name.replace('/', '-')
            tokenizer = args.tokenizer_type
            prefix = dataset_name + '_' + model_name + '-' + tokenizer + '-'
            histogram_file = os.path.join(self.dataset_loader.get_dataset_dir(), prefix + 'histogram.txt')

            with open(histogram_file, mode='w', encoding='utf-8') as f:
                f.write("Average len:{:.4f}".format(avg_len) + '\n')
                f.write("Max len:" + str(max_len) + '\n')
                f.write('length - count' + '\n')
                for (length, count) in cnt:
                    # print()
                    f.write(str(length) + ' - ' + str(count) + '\n')

            logger.info(f"Max tokens len:{max_len}")
            logger.info(f"Avg tokens len:{avg_len}")

            tokens_histogram_path = os.path.join(self.dataset_loader.get_dataset_dir(), prefix + 'tokens_histogram.png')
            logger.info(f"Tokens histogram image saved to:{tokens_histogram_path}")
            # fig, ax = plt.subplots()

            # based on https://github.com/mwaskom/seaborn/issues/479#issuecomment-333304727
            plt.figure()  # it resets the plot

            # Plotting hist without kde
            ax = sns.distplot(token_lens, kde=False, color='blue')

            # Creating another Y axis
            second_ax = ax.twinx()

            # Plotting kde without hist on the second Y axis
            sns.distplot(token_lens, ax=second_ax, kde=True, hist=False)
            second_ax.set_yticks([])

            # Removing Y ticks from the second axis
            # ax.set(xlabel=f"Token count, max len:{max_len}", ylabel='Frequency')
            ax.set(xlabel=f"Subword Token Counts", ylabel='Frequency')
            # plt.xlabel(f"Token count, max len:{max_len}")
            # plt.ylabel("Frequency")
            # plt.show()

            x_max = 1024
            plt.xticks(np.arange(0, x_max + 1, 100))
            plt.xlim([0, x_max])

            sns.despine(top=True, right=True, left=False, bottom=False)

            plt.savefig(tokens_histogram_path, dpi=400)
            plt.savefig(tokens_histogram_path + ".pdf")
            plt.figure()

            print()


def fine_tune_torch(args):
    logger.info("Initializing torch tuner")
    tuner = TorchTuner(args)

    if args.task == 'CAT':
        tuner.fine_tune_cat_task(args)
    elif args.task == 'TERM':
        tuner.fine_tune_term_task(args)
    elif args.task == 'SRL':
        tuner.fine_tune_cat_task(args)
    else:
        raise Exception("Unknown type of task")
    pass

    logger.info("Deleting tuner")
    del tuner
    gc.collect()
    # Clean memory
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    logger.info("--------")
    print_gpu_info()
    logger.info(70 * "---")
    logger.info("Tuner deleted")


#

def print_gpu_info():
    try:
        logger.info(f"GPU first device name:{torch.cuda.get_device_name(0)}")
        # t = torch.cuda.get_device_properties(0).total_memory
        # c = torch.cuda.memory_cached(0)
        # a = torch.cuda.memory_allocated(0)
        # f = c - a  # free inside cache
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        t = info.total
        f = info.free
        a = info.used
        logger.info(f'GPU 0 Memory total    : {int(t / (1024.0 * 1024.0))} MiB')
        logger.info(f'GPU 0 Memory free     : {int(f / (1024.0 * 1024.0))} MiB')
        logger.info(f'GPU 0 Memory used     : {int(a / (1024.0 * 1024.0))} MiB')

        logger.info(f"GPU first device name:{torch.cuda.get_device_name(1)}")
        # t = torch.cuda.get_device_properties(0).total_memory
        # c = torch.cuda.memory_cached(0)
        # a = torch.cuda.memory_allocated(0)
        # f = c - a  # free inside cache
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(1)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        t = info.total
        f = info.free
        a = info.used
        logger.info(f'GPU 0 Memory total    : {int(t / (1024.0 * 1024.0))} MiB')
        logger.info(f'GPU 0 Memory free     : {int(f / (1024.0 * 1024.0))} MiB')
        logger.info(f'GPU 0 Memory used     : {int(a / (1024.0 * 1024.0))} MiB')
        # print(f'cached   : {c/(1024.0*1024.0)}')
    except Exception as e:
        logger.info("Exception during:" + str(e))
