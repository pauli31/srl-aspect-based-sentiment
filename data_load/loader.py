import logging
import math
import os

import pandas as pd

from config import MODES, ENGLISH, CZECH, \
    SEMEVAL_2014_CS_GENERATED_LABELS_BINARY_MAP_INDICES, SEMEVAL_2014_EN_GENERATED_LABELS_BINARY_MAP_INDICES, \
    GENERATED_DATASET_DIR_SEMEVAL_PAIR_EN, GENERATED_DATASET_DIR_SEMEVAL_PAIR_CS, \
    GENERATED_DATASET_DIR_SEMEVAL_PAIR_CS_KFOLD, MONOLINGUAL_MODE, SRL_DATA_DIR
# import utils_srl
from utils import utils_srl

logger = logging.getLogger(__name__)



class ABSADataLoader(object):
    def __init__(self, args, source_lang, target_lang=None):
        self.max_train_data = args.max_train_data
        self.args = args

        if args.lang_mode not in MODES:
            raise ValueError("Unknown mode: " + str(args.lang_mode))
        else:
            self.lang_mode = args.lang_mode

        # Source language of the dataset
        self.source_lang = source_lang

        # Target language of the dataset only, if lang_mode is cross_lingual
        self.target_lang = target_lang

        self.train_data = None
        self.test_data = None
        self.dev_data = None
        self.datasets = []

    def get_lang_mode(self):
        return self.lang_mode

    def get_source_lang(self):
        return self.source_lang

    def get_target_lang(self):
        return self.target_lang

    def get_dev_data(self):
        if self.dev_data is None:
            self.load_data()
        return self.dev_data

    def get_test_data(self):
        if self.test_data is None:
            self.load_data()
        return self.test_data

    def get_train_data(self):
        if self.train_data is None:
            self.load_data()

        data_to_return = self.get_cutted_train_data(self.train_data)

        return data_to_return

    def get_train_dev_data(self):
        if self.train_data is None:
            self.load_data()
        if self.dev_data is None:
            self.load_data()

        df = pd.concat([self.train_data, self.dev_data], axis=0)
        df.reset_index(drop=True, inplace=True)

        data_to_return = self.get_cutted_train_data(df)

        return data_to_return

    def get_cutted_train_data(self, data_df):

        data_to_return = data_df
        total_size = len(data_df)
        logger.info("The size of training dataset is:" + str(total_size))
        logger.info("Applying cutting to train data with value:" + str(self.max_train_data))

        new_size = -1
        if self.max_train_data <= 0:
            logger.info("No cutting is performed")
        elif 0 < self.max_train_data <= 1:
            logger.info("Cutting in percentages")
            new_size = total_size * self.max_train_data
            new_size = math.ceil(new_size)
        elif self.max_train_data > 1:
            logger.info("Cutting in absolute numbers")
            new_size = self.max_train_data
            new_size = math.ceil(new_size)
        else:
            raise Exception("Unknown value for max_train_data, the value:" + str(self.max_train_data))

        logger.info("New size is:" + str(new_size))
        if new_size > 1:
            data_to_return = data_to_return.head(new_size)
            data_to_return.reset_index(drop=True, inplace=True)

        return data_to_return

    def load_entire_dataset(self):
        raise NotImplementedError()

    def load_data(self):
        """Loads all data"""
        raise NotImplementedError()

    def get_dataset_dir(self):
        """Returns dir of the dataset"""
        raise NotImplementedError()

    def get_class_num(self):
        raise NotImplementedError()

    def get_aspect_classes_num(self):
        raise NotImplementedError()

    def get_sentiment_classes_num(self):
        raise NotImplementedError()

    def get_generated_sentiment_labels_mapping(self):
        raise NotImplementedError()

    def get_classes(self, eval=False):
        """Returns possible clases as numbers"""
        raise NotImplementedError()

    def get_class_names(self):
        """
        Returns possible names of classes
        Indices corresponds to the ones returned by method get_classes
        """
        raise NotImplementedError()


    def get_text4label(self, label):
        """
        Returns text for numerical label
        :param label: numerical label
        :return text for numerical label
        """
        raise NotImplementedError()

    def get_label4text(self, text_label):
        """
        Returns numerical label for text label
        :param text_label: text label
        :return:
        """
        raise NotImplementedError()

    def get_label4text_evaluation(self, text_label):
        raise NotImplementedError()


    # treba pridat sloupec label_text tam dat co je v label a ve sloupci label namapovat
    def map_labels(self, data_df):
        if self.args.task == 'CAT':
            if self.args.solution_type_cat == 'NLI_M' or self.args.solution_type_cat == 'QA_M':
                label_names = data_df.label
                labels = data_df.label
                labels = labels.apply(lambda x: self.get_label4text(x))
                data_df['label_text'] = label_names
                data_df['label'] = labels
            else:
                labels_names = data_df.label
                labels_names = labels_names.apply(lambda x: self.get_text4label(x))
                data_df['label_text'] = labels_names
        else:
            raise Exception("Unsupported task:" + self.args.task)

        return data_df


class End2EndSRLDataLoader(object):
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.dev_data = None
        self.data_dir = os.path.join(SRL_DATA_DIR, args.dataset_lang)
        self.labels = utils_srl.get_labels(os.path.join(self.data_dir, "labels.txt"))

    def load_data(self):
        self.test_data = self._load_data_part("test")
        self.dev_data = self._load_data_part("dev")
        self.train_data = self._load_data_part("train")

    def _load_data_part(self, split):
        if self.args.dataset_lang == "cs":
            return utils_srl.read_examples_from_file_e2e(self.data_dir, split)
        else:
            return utils_srl.read_examples_from_ontonotes(self.data_dir, split)

    def get_dev_data(self):
        if self.dev_data is None:
            self.dev_data = self._load_data_part("dev")
        return self.dev_data

    def get_test_data(self):
        if self.test_data is None:
            self.test_data = self._load_data_part("test")
        return self.test_data

    def get_train_data(self):
        if self.train_data is None:
            self.train_data = self._load_data_part("train")

        return self.train_data

    def get_train_dev_data(self):
        if self.train_data is None:
            self.train_data = self._load_data_part("train")
        if self.dev_data is None:
            self.test_data = self._load_data_part("test")

        df = pd.concat([self.train_data, self.dev_data], axis=0)
        df.reset_index(drop=True, inplace=True)

        return df

    def get_class_num_srl(self):
        # Ondra rika 125 magicke cislo
        return len(self.labels)
        # raise NotImplementedError()

    def get_class_num(self):
        return self.get_class_num_srl()

    def load_entire_dataset(self):
        if self.train_data is None:
            self.load_data()

        list_all = []
        list_all.extend(self.train_data, self.test_data, self.dev_data)
        return list_all

    def get_lang_mode(self):
        return MONOLINGUAL_MODE



class CSSRLDataLoader(object):
    def __init__(self, args, end2end=False):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.dev_data = None
        self.end2end = end2end
        self.data_dir = os.path.join(SRL_DATA_DIR, args.dataset_lang)

    def load_data(self):
        self.test_data = utils_srl.read_examples_from_file(self.data_dir, "test")
        self.dev_data = utils_srl.read_examples_from_file(self.data_dir, "dev")
        self.train_data = utils_srl.read_examples_from_file(self.data_dir, "train")

    def get_dev_data(self):
        if self.dev_data is None:
            self.load_data()
        return self.dev_data

    def get_test_data(self):
        if self.test_data is None:
            self.load_data()
        return self.test_data

    def get_train_data(self):
        if self.train_data is None:
            self.load_data()

        return self.train_data

    def get_train_dev_data(self):
        if self.train_data is None:
            self.load_data()
        if self.dev_data is None:
            self.load_data()

        df = pd.concat([self.train_data, self.dev_data], axis=0)
        df.reset_index(drop=True, inplace=True)

        return df

    def get_class_num_srl(self):
        # Ondra rika 125 magicke cislo
        return 125
        # raise NotImplementedError()

    def get_class_num(self):
        return self.get_class_num_srl()

    def load_entire_dataset(self):
        if self.train_data is None:
            self.load_data()

        list_all = []
        list_all.extend(self.train_data, self.test_data, self.dev_data)
        return list_all

    def get_lang_mode(self):
        return MONOLINGUAL_MODE


class EN_SRL_ABSADataLoader(object):
    def __init__(self, args):
        self.args = args
        self.absa_loader = ENSemEval2014(args)
        args.dataset_lang = 'en'
        self.srl_loader = End2EndSRLDataLoader(args)

    def load_data(self):
        """Loads all data"""
        self.absa_loader.load_data()
        self.srl_loader.load_data()

    def get_dev_data(self):
        absa_dev = self.absa_loader.get_dev_data()
        srl_dev = self.srl_loader.get_dev_data()
        return (absa_dev, srl_dev)

    def get_test_data(self):
        absa_test = self.absa_loader.get_test_data()
        srl_test = self.srl_loader.get_test_data()
        return (absa_test, srl_test)

    def get_train_data(self):
        absa_train = self.absa_loader.get_train_data()
        srl_train = self.srl_loader.get_train_data()
        return (absa_train, srl_train)

    def get_dataset_dir(self):
        """Returns dir of the dataset"""
        raise NotImplementedError()

    def get_class_num_srl(self):
        return self.srl_loader.get_class_num_srl()

    def get_class_num(self):
        return self.absa_loader.get_class_num()

    def get_aspect_classes_num(self):
        return self.absa_loader.get_aspect_classes_num()

    def get_sentiment_classes_num(self):
        return self.absa_loader.get_sentiment_classes_num()

    def get_generated_sentiment_labels_mapping(self):
        return self.absa_loader.get_generated_sentiment_labels_mapping()

    def get_classes(self, eval=False):
        """Returns possible clases as numbers"""
        return self.absa_loader.get_classes(eval=eval)

    def get_class_names(self):
        """
        Returns possible names of classes
        Indices corresponds to the ones returned by method get_classes
        """
        return self.absa_loader.get_class_names()

    def get_text4label(self, label):
        """
        Returns text for numerical label
        :param label: numerical label
        :return text for numerical label
        """
        return self.absa_loader.get_text4label(label)

    def get_label4text(self, text_label):
        """
        Returns numerical label for text label
        :param text_label: text label
        :return:
        """
        return self.absa_loader.get_label4text(text_label)

    def get_label4text_evaluation(self, text_label):
        return self.absa_loader.get_label4text_evaluation(text_label)

    def get_text4label_evaluation(self, label):
        return self.absa_loader.get_text4label_evaluation(label)

    def get_lang_mode(self):
        return MONOLINGUAL_MODE

class CS_SRL_ABSADataLoader(object):
    def __init__(self, args):
        self.args = args
        self.absa_loader = CSSemEval2014(args)
        # self.srl_loader = CSSRLDataLoader(args)
        self.srl_loader = End2EndSRLDataLoader(args)

    def load_data(self):
        """Loads all data"""
        self.absa_loader.load_data()
        self.srl_loader.load_data()

    def get_dev_data(self):
        absa_dev = self.absa_loader.get_dev_data()
        srl_dev = self.srl_loader.get_dev_data()
        return (absa_dev, srl_dev)

    def get_test_data(self):
        absa_test = self.absa_loader.get_test_data()
        srl_test = self.srl_loader.get_test_data()
        return (absa_test, srl_test)

    def get_train_data(self):
        absa_train = self.absa_loader.get_train_data()
        srl_train = self.srl_loader.get_train_data()
        return (absa_train, srl_train)

    def get_dataset_dir(self):
        """Returns dir of the dataset"""
        raise NotImplementedError()

    def get_class_num_srl(self):
        return self.srl_loader.get_class_num_srl()

    def get_class_num(self):
        return self.absa_loader.get_class_num()

    def get_aspect_classes_num(self):
        return self.absa_loader.get_aspect_classes_num()

    def get_sentiment_classes_num(self):
        return self.absa_loader.get_sentiment_classes_num()

    def get_generated_sentiment_labels_mapping(self):
        return self.absa_loader.get_generated_sentiment_labels_mapping()

    def get_classes(self, eval=False):
        """Returns possible clases as numbers"""
        return self.absa_loader.get_classes(eval=eval)

    def get_class_names(self):
        """
        Returns possible names of classes
        Indices corresponds to the ones returned by method get_classes
        """
        return self.absa_loader.get_class_names()

    def get_text4label(self, label):
        """
        Returns text for numerical label
        :param label: numerical label
        :return text for numerical label
        """
        return self.absa_loader.get_text4label(label)

    def get_label4text(self, text_label):
        """
        Returns numerical label for text label
        :param text_label: text label
        :return:
        """
        return self.absa_loader.get_label4text(text_label)

    def get_label4text_evaluation(self, text_label):
        return self.absa_loader.get_label4text_evaluation(text_label)

    def get_text4label_evaluation(self, label):
        return self.absa_loader.get_text4label_evaluation(label)

    def get_lang_mode(self):
        return MONOLINGUAL_MODE


class ENSemEval2014(ABSADataLoader):
    def __init__(self, args):
        super().__init__(args, ENGLISH)
        if args.task == 'CAT':
            train_file_name = "EN_train_" + args.solution_type_cat + ".tsv"
            dev_file_name = "EN_dev_" + args.solution_type_cat + ".tsv"
            test_file_name = "EN_test_" + args.solution_type_cat + ".tsv"
        elif args.task == 'TERM':
            raise Exception("Not implemented yet!")
        else:
            raise Exception("Unsupported task type")

        self.train_path = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_PAIR_EN, train_file_name)
        self.dev_path = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_PAIR_EN, dev_file_name)
        self.test_path = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_PAIR_EN, test_file_name)

    def load_data(self):
        # it cannot be shuffled since wee need to keep the order for dev data for evaluation

        # load data
        self.train_data = self.map_labels(pd.read_csv(self.train_path, sep='\t'))
        self.dev_data = self.map_labels(pd.read_csv(self.dev_path, sep='\t'))
        self.test_data = self.map_labels(pd.read_csv(self.test_path, sep='\t'))

    def load_entire_dataset(self):
        train_data = pd.read_csv(self.train_path, sep='\t')
        test_data = pd.read_csv(self.test_path, sep='\t')
        dev_data = pd.read_csv(self.dev_path, sep='\t')

        df = pd.concat([train_data, dev_data, test_data])
        df.reset_index(drop=True, inplace=True)

        df = self.map_labels(df)

        return df

    def get_dataset_dir(self):
        raise NotImplementedError("Not implemented yet, do it now")

    def get_class_num(self):
        task = self.args.task
        if task == 'CAT':
            type = self.args.solution_type_cat
            if type == 'QA_B' or type == 'NLI_B':
                num_classes = 2
            elif type == 'NLI_M' or type == 'QA_M':
                num_classes = 5
            else:
                raise Exception("Unknown type:" + str(type))
        elif task == 'TERM':
            raise NotImplementedError("Not implemented ")
        else:
            raise ValueError("Unknown task type")

        return num_classes

    def get_aspect_classes_num(self):
        return 5

    def get_classes(self, eval=False):
        task = self.args.task
        if task == 'CAT':
            type = self.args.solution_type_cat
            if (type == 'QA_B' or type == 'NLI_B') and not eval:
                classes = [0, 1]
            elif (type == 'NLI_M' or type == 'QA_M') or eval:
                classes = [0, 1, 2, 3, 4]
            else:
                raise Exception("Unknown type:" + str(type))
        elif task == 'TERM':
            raise NotImplementedError("Not implemented ")
        else:
            raise ValueError("Unknown task type")

        return classes

    def get_class_names(self):
        """
        Returns possible names of classes
        Indices corresponds to the ones returned by method get_classes
        """
        task = self.args.task
        if task == 'CAT':
            type = self.args.solution_type_cat
            if type == 'QA_B' or type == 'NLI_B':
                class_names = ['negative', 'positive']
            elif type == 'NLI_M' or type == 'QA_M':
                class_names = ['negative', 'positive', 'neutral', 'conflict', 'none']
            else:
                raise Exception("Unknown type:" + str(type))
        elif task == 'TERM':
            raise NotImplementedError("Not implemented ")
        else:
            raise ValueError("Unknown task type")

        return class_names

    def get_sentiment_classes_num(self):
        return 5

    def get_generated_sentiment_labels_mapping(self):
        return SEMEVAL_2014_EN_GENERATED_LABELS_BINARY_MAP_INDICES

    def get_text4label(self, label):
        """
        Returns text for numerical label
        :param label: numerical label
        :return:
        """
        task = self.args.task
        if task == 'CAT':
            type = self.args.solution_type_cat
            if type == 'QA_B' or type == 'NLI_B':
                if label == 0:
                    ret = 'negative'
                elif label == 1:
                    ret = 'positive'
                else:
                    raise Exception("Unknown label:" + str(label))
            elif type == 'NLI_M' or type == 'QA_M':
                if label == 0:
                    ret = 'negative'
                elif label == 1:
                    ret = 'positive'
                elif label == 2:
                    ret = 'neutral'
                elif label == 3:
                    ret = 'conflict'
                elif label == 4:
                    ret = 'none'
                else:
                    raise Exception("Unknown label:" + str(label))
            else:
                raise Exception("Unknown type:" + str(type))
        elif task == 'TERM':
            raise NotImplementedError("Not implemented ")
        else:
            raise ValueError("Unknown task type")

        return ret


    def get_label4text(self, text_label):
        """
        Returns numerical label for text label
        :param text_label: text label
        :return:
        """
        task = self.args.task
        if task == 'CAT':
            type = self.args.solution_type_cat
            if type == 'QA_B' or type == 'NLI_B':
                if text_label == 'negative':
                    ret = 0
                elif text_label == 'positive':
                    ret = 1
                else:
                    raise Exception("Unknown label:" + str(text_label))
            elif type == 'NLI_M' or type == 'QA_M':
                if text_label == 'negative':
                    ret = 0
                elif text_label == 'positive':
                    ret = 1
                elif text_label == 'neutral':
                    ret = 2
                elif text_label == 'conflict':
                    ret = 3
                elif text_label == 'none':
                    ret = 4
                else:
                    raise Exception("Unknown label:" + str(text_label))
            else:
                raise Exception("Unknown type:" + str(type))
        elif task == 'TERM':
            raise NotImplementedError("Not implemented ")
        else:
            raise ValueError("Unknown task type")

        return ret
    
    def get_label4text_evaluation(self, text_label):
        task = self.args.task
        if task == 'CAT':
            type = self.args.solution_type_cat
            if type == 'QA_B' or type == 'NLI_B' or type == 'NLI_M' or type == 'QA_M':
                if text_label == 'negative':
                    ret = 0
                elif text_label == 'positive':
                    ret = 1
                elif text_label == 'neutral':
                    ret = 2
                elif text_label == 'conflict':
                    ret = 3
                elif text_label == 'none':
                    ret = 4
                else:
                    raise Exception("Unknown label:" + str(text_label))
            else:
                raise Exception("Unknown type:" + str(type))
        elif task == 'TERM':
            raise NotImplementedError("Not implemented ")
        else:
            raise ValueError("Unknown task type")

        return ret

    def get_text4label_evaluation(self, label):
        task = self.args.task
        if task == 'CAT':
            type = self.args.solution_type_cat
            if type == 'QA_B' or type == 'NLI_B' or type == 'NLI_M' or type == 'QA_M':
                if label == 0:
                    ret = 'negative'
                elif label == 1:
                    ret = 'positive'
                elif label == 2:
                    ret = 'neutral'
                elif label == 3:
                    ret = 'conflict'
                elif label == 4:
                    ret = 'none'
                else:
                    raise Exception("Unknown label:" + str(label))
            else:
                raise Exception("Unknown type:" + str(type))
        elif task == 'TERM':
            raise NotImplementedError("Not implemented ")
        else:
            raise ValueError("Unknown task type")

        return ret




class CSSemEval2014(ABSADataLoader):
    def __init__(self, args):
        super().__init__(args, CZECH)
        if args.task == 'CAT':
            train_file_name = "CS_train_" + args.solution_type_cat + ".tsv"
            dev_file_name = "CS_dev_" + args.solution_type_cat + ".tsv"
            test_file_name = "CS_test_" + args.solution_type_cat + ".tsv"

        elif args.task == 'TERM':
            raise Exception("Not implemented yet!")
        else:
            raise Exception("Unsupported task type")

        self.train_path = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_PAIR_CS, train_file_name)
        self.dev_path = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_PAIR_CS, dev_file_name)
        self.test_path = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_PAIR_CS, test_file_name)

    def load_data(self):
        # load data
        self.train_data = self.map_labels(pd.read_csv(self.train_path, sep='\t'))
        self.dev_data = self.map_labels(pd.read_csv(self.dev_path, sep='\t'))
        self.test_data = self.map_labels(pd.read_csv(self.test_path, sep='\t'))

    def load_entire_dataset(self):
        train_data = pd.read_csv(self.train_path, sep='\t')
        test_data = pd.read_csv(self.test_path, sep='\t')
        dev_data = pd.read_csv(self.dev_path, sep='\t')

        df = pd.concat([train_data, dev_data, test_data])
        df.reset_index(drop=True, inplace=True)

        df = self.map_labels(df)

        return df

    def get_dataset_dir(self):
        raise NotImplementedError("Not implemented yet, do it now")

    def get_class_num(self):
        """
        Returns number of classes based on the given type of task and solution used
        :return:
        """
        task = self.args.task
        if task == 'CAT':
            type = self.args.solution_type_cat
            if type == 'QA_B' or type == 'NLI_B':
                num_classes = 2
            elif type == 'NLI_M' or type == 'QA_M':
                num_classes = 4
            else:
                raise Exception("Unknown type:" + str(type))
        elif task == 'TERM':
            raise NotImplementedError("Not implemented ")
        else:
            raise ValueError("Unknown task type")

        return num_classes

    def get_aspect_classes_num(self):
        return 5

    # nebo [0,1,2,4]
    def get_classes(self, eval=False):
        task = self.args.task
        if task == 'CAT':
            type = self.args.solution_type_cat
            if (type == 'QA_B' or type == 'NLI_B') and not eval:
                classes = [0, 1]
            elif (type == 'NLI_M' or type == 'QA_M') or eval:
                classes = [0, 1, 2, 3]
            else:
                raise Exception("Unknown type:" + str(type))
        elif task == 'TERM':
            raise NotImplementedError("Not implemented ")
        else:
            raise ValueError("Unknown task type")

        return classes


    def get_class_names(self):
        """
        Returns possible names of classes
        Indices corresponds to the ones returned by method get_classes
        """
        task = self.args.task
        if task == 'CAT':
            type = self.args.solution_type_cat
            if type == 'QA_B' or type == 'NLI_B':
                class_names = ['negative', 'positive']
            elif type == 'NLI_M' or type == 'QA_M':
                class_names = ['negative', 'positive', 'neutral', 'none']
            else:
                raise Exception("Unknown type:" + str(type))
        elif task == 'TERM':
            raise NotImplementedError("Not implemented ")
        else:
            raise ValueError("Unknown task type")

        return class_names

    def get_sentiment_classes_num(self):
        return 4

    def get_generated_sentiment_labels_mapping(self):
        return SEMEVAL_2014_CS_GENERATED_LABELS_BINARY_MAP_INDICES


    def get_text4label(self, label):
        """
        Returns text for numerical label
        :param label: numerical label
        :return:
        """
        task = self.args.task
        if task == 'CAT':
            type = self.args.solution_type_cat
            if type == 'QA_B' or type == 'NLI_B':
                if label == 0:
                    ret = 'negative'
                elif label == 1:
                    ret = 'positive'
                else:
                    raise Exception("Unknown label:" + str(label))
            elif type == 'NLI_M' or type == 'QA_M':
                if label == 0:
                    ret = 'negative'
                elif label == 1:
                    ret = 'positive'
                elif label == 2:
                    ret = 'neutral'
                elif label == 3:
                    ret = 'none'
                else:
                    raise Exception("Unknown label:" + str(label))
            else:
                raise Exception("Unknown type:" + str(type))
        elif task == 'TERM':
            raise NotImplementedError("Not implemented ")
        else:
            raise ValueError("Unknown task type")

        return ret


    def get_label4text(self, text_label):
        """
        Returns numerical label for text label
        :param text_label: text label
        :return:
        """
        task = self.args.task
        if task == 'CAT':
            type = self.args.solution_type_cat
            if type == 'QA_B' or type == 'NLI_B':
                if text_label == 'negative':
                    ret = 0
                elif text_label == 'positive':
                    ret = 1
                else:
                    raise Exception("Unknown label:" + str(text_label))
            elif type == 'NLI_M' or type == 'QA_M':
                if text_label == 'negative':
                    ret = 0
                elif text_label == 'positive':
                    ret = 1
                elif text_label == 'neutral':
                    ret = 2
                elif text_label == 'none':
                    ret = 3
                else:
                    raise Exception("Unknown label:" + str(text_label))
            else:
                raise Exception("Unknown type:" + str(type))
        elif task == 'TERM':
            raise NotImplementedError("Not implemented ")
        else:
            raise ValueError("Unknown task type")

        return ret


    def get_label4text_evaluation(self, text_label):
        task = self.args.task
        if task == 'CAT':
            type = self.args.solution_type_cat
            if type == 'QA_B' or type == 'NLI_B' or type == 'NLI_M' or type == 'QA_M':
                if text_label == 'negative':
                    ret = 0
                elif text_label == 'positive':
                    ret = 1
                elif text_label == 'neutral':
                    ret = 2
                elif text_label == 'none':
                    ret = 3
                else:
                    raise Exception("Unknown label:" + str(text_label))
            else:
                raise Exception("Unknown type:" + str(type))
        elif task == 'TERM':
            raise NotImplementedError("Not implemented ")
        else:
            raise ValueError("Unknown task type")

        return ret

    def get_text4label_evaluation(self, label):
        task = self.args.task
        if task == 'CAT':
            type = self.args.solution_type_cat
            if type == 'QA_B' or type == 'NLI_B' or type == 'NLI_M' or type == 'QA_M':
                if label == 0:
                    ret = 'negative'
                elif label == 1:
                    ret = 'positive'
                elif label == 2:
                    ret = 'neutral'
                elif label == 3:
                    ret = 'none'
                else:
                    raise Exception("Unknown label:" + str(label))
            else:
                raise Exception("Unknown type:" + str(type))
        elif task == 'TERM':
            raise NotImplementedError("Not implemented ")
        else:
            raise ValueError("Unknown task type")

        return ret


class CSSemEval2014_Cross_Validation(CSSemEval2014):
    def __init__(self, args):
        super().__init__(args)
        self.num_folds = 10
        self.current_fold = args.current_fold


        if args.task == 'CAT':
            train_file_names = []
            dev_file_names = []
            test_file_names = []

            for i in range(self.num_folds):
                train_file_names.append("CS_train_kfold_" + str(i) + "_" + args.solution_type_cat + ".tsv")
                dev_file_names.append("CS_dev_kfold_" + str(i) + "_" + args.solution_type_cat + ".tsv")
                test_file_names.append("CS_test_kfold_" + str(i) + "_" + args.solution_type_cat + ".tsv")

        elif args.task == 'TERM':
            raise Exception("Not implemented yet!")
        else:
            raise Exception("Unsupported task type")

        self.train_paths = []
        self.dev_paths = []
        self.test_paths = []

        for i in range(self.num_folds):
            self.train_paths.append(os.path.join(GENERATED_DATASET_DIR_SEMEVAL_PAIR_CS_KFOLD, train_file_names[i]))
            self.dev_paths.append(os.path.join(GENERATED_DATASET_DIR_SEMEVAL_PAIR_CS_KFOLD, dev_file_names[i]))
            self.test_paths.append(os.path.join(GENERATED_DATASET_DIR_SEMEVAL_PAIR_CS_KFOLD, test_file_names[i]))



    def load_data(self):
        # load data
        self.train_data = []
        self.dev_data = []
        self.test_data = []

        for i in range(self.num_folds):
            self.train_data.append(self.map_labels(pd.read_csv(self.train_paths[i], sep='\t')))
            self.dev_data.append(self.map_labels(pd.read_csv(self.dev_paths[i], sep='\t')))
            self.test_data.append(self.map_labels(pd.read_csv(self.test_paths[i], sep='\t')))

    def get_train_dev_data(self):
        if self.train_data is None:
            self.load_data()
        if self.dev_data is None:
            self.load_data()

        train_data = self.train_data[self.current_fold]
        dev_data = self.dev_data[self.current_fold]

        df = pd.concat([train_data, dev_data], axis=0)
        df.reset_index(drop=True, inplace=True)

        return df

    def get_test_data(self):
        if self.test_data is None:
            self.load_data()
        return self.test_data[self.current_fold]

    def get_dev_data(self):
        raise Exception("Only train-dev data can be loaded in this setting")

    def get_train_data(self):
        raise Exception("Only train-dev data can be loaded in this setting")



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$ Cross-lingual $$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

class CrossLingualABSADataset(ABSADataLoader):
    """
    As dev data the subclass dataset MUST return the source language data
    As test data the subclass dataset MUST return the target language data
    """

    def __init__(self, args, source_lang, target_lang):
        super().__init__(args, source_lang, target_lang)

    def get_source_lang_dev_data(self):
        """Returns the dev data for the source language"""
        self.get_dev_data()

    def get_target_lang_dev_data(self):
        """Returns the dev data for the target language"""
        raise NotImplementedError()

    def load_entire_source_lang_dataset(self):
        raise NotImplementedError()

    def load_entire_target_lang_dataset(self):
        raise NotImplementedError()

    def load_entire_dataset(self):
        df_source = self.load_entire_source_lang_dataset()
        df_target = self.load_entire_target_lang_dataset()

        df = pd.concat([df_source, df_target], axis=0)
        df.reset_index(drop=True, inplace=True)
        return df


DATASET_LOADERS = {
    "semeval2014_en": ENSemEval2014,
    "semeval2014_cs": CSSemEval2014,
    "semeval2014_cs-cross_val": CSSemEval2014_Cross_Validation,
    "cs_absa_srl_dataset": CS_SRL_ABSADataLoader,
    "en_absa_srl_dataset": EN_SRL_ABSADataLoader,
    "cs_srl_e2e": End2EndSRLDataLoader,
    "cs_srl": CSSRLDataLoader
}