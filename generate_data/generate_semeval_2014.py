# modified version of the https://github.com/HSLCY/ABSA-BERT-pair
import logging
import os
import sys
from pathlib import Path
import numpy as np

import jsonpickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle

from data_utils import parse_semeval_2014

# https://stackoverflow.com/questions/43728431/relative-imports-modulenotfounderror-no-module-named-x
# need to set because the config is in the parent directory

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from args_utils import build_parser_generating

from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT, GENERATED_DATASET_DIR_SEMEVAL_PAIR, \
    EN_SEMEVAL_2014_RAW_DATASET_TEST, SEMEVAL_2014_ASPECT_CATEGORIES, CS_SEMEVAL_2014_RAW_DATASET, \
    EN_ASPECT_CATEGORIES_QUESTIONS_STRINGS, CS_ASPECT_CATEGORIES_QUESTIONS_STRINGS, EN_SEMEVAL_2014_RAW_DATASET_TRAIN, \
    EN_SEMEVAL_2014_LABELS_GENERATING_MAPING, CS_SEMEVAL_2014_LABELS_GENERATING_MAPING, \
    GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA, SEMEVAL_2014_CS_LABELS_GENERATING, SEMEVAL_2014_CS_LABELS_CUSTOM_QA, \
    SEMEVAL_2014_EN_LABELS_GENERATING, SEMEVAL_2014_EN_LABELS_CUSTOM_QA, RANDOM_SEED, \
    CS_SEMEVAL_SPLIT_DATA_TRAIN, CS_SEMEVAL_SPLIT_DATA_TEST, CS_SEMEVAL_SPLIT_DATA_DEV, TEST_SIZE_CS, DEV_SIZE_EN, \
    DEV_SIZE_CS, EN_SEMEVAL_SPLIT_DATA_TRAIN, EN_SEMEVAL_SPLIT_DATA_TEST, EN_SEMEVAL_SPLIT_DATA_DEV, \
    GENERATED_DATASET_DIR_SEMEVAL_PAIR_EN, GENERATED_DATASET_DIR_SEMEVAL_PAIR_CS, \
    GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA_EN, GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA_CS, CS_NUMBER_FOLDS, \
    GENERATED_DATASET_DIR_SEMEVAL_SPLIT_CS_KFOLD, GENERATED_DATASET_DIR_SEMEVAL_PAIR_CS_KFOLD, \
    GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA_CS_KFOLD

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)



def main():
    split_data_czech()
    parser = build_parser_generating()
    args = parser.parse_args()

    Path(GENERATED_DATASET_DIR_SEMEVAL_PAIR).mkdir(parents=True, exist_ok=True)
    Path(GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA).mkdir(parents=True, exist_ok=True)

    if args.split_data is True:
        split_data_czech()
        split_data_english()

    if args.source == 'semeval2014':
        generate_semeval_2014(args)
    elif args.source == 'semeval2016':
        raise NotImplementedError("Not implemented Semeval2016, implement it!")
    elif args.source == 'sentihood':
        raise NotImplementedError("Not implemented Sentihood, implement it!")
    else:
        raise Exception("Unknown source:" + str(args.source))


def split_data_english():
    test_data = parse_semeval_2014(EN_SEMEVAL_2014_RAW_DATASET_TEST)

    aspect_examples = parse_semeval_2014(EN_SEMEVAL_2014_RAW_DATASET_TRAIN)
    aspect_examples = shuffle(aspect_examples, random_state=RANDOM_SEED)
    train_data, dev_data = train_test_split(aspect_examples, test_size=DEV_SIZE_EN, random_state=RANDOM_SEED)

    save_json_file(train_data, EN_SEMEVAL_SPLIT_DATA_TRAIN)
    save_json_file(test_data, EN_SEMEVAL_SPLIT_DATA_TEST)
    save_json_file(dev_data, EN_SEMEVAL_SPLIT_DATA_DEV)

    print("EN Data split train size:" + str(len(train_data)))
    print("EN Data split test size:" + str(len(test_data)))
    print("EN Data split dev size:" + str(len(dev_data)))


def split_data_czech():
    aspect_examples = parse_semeval_2014(CS_SEMEVAL_2014_RAW_DATASET)
    # shuffle it
    aspect_examples = shuffle(aspect_examples, random_state=RANDOM_SEED)

    train_data, test_data = train_test_split(aspect_examples, test_size=TEST_SIZE_CS, random_state=RANDOM_SEED)
    train_data, dev_data = train_test_split(train_data, test_size=DEV_SIZE_CS, random_state=RANDOM_SEED)

    save_json_file(train_data, CS_SEMEVAL_SPLIT_DATA_TRAIN)
    save_json_file(test_data, CS_SEMEVAL_SPLIT_DATA_TEST)
    save_json_file(dev_data, CS_SEMEVAL_SPLIT_DATA_DEV)

    print("CS Data split train size:" + str(len(train_data)))
    print("CS Data split test size:" + str(len(test_data)))
    print("CS Data split dev size:" + str(len(dev_data)))

    aspect_examples = np.array(aspect_examples)
    kf = KFold(n_splits=CS_NUMBER_FOLDS)
    for i, (train_idx, test_idx) in enumerate(kf.split(aspect_examples)):
        print("Generating fold number:" + str(i))
        test = aspect_examples[test_idx]
        train = aspect_examples[train_idx]

        train, dev = train_test_split(train, test_size=DEV_SIZE_CS, random_state=RANDOM_SEED)
        save_path_train = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_SPLIT_CS_KFOLD, 'CzechABSA-v2_format_SemEval2014-TRAIN-fold_' + str(i) + '.json')
        save_path_test = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_SPLIT_CS_KFOLD, 'CzechABSA-v2_format_SemEval2014-TEST-fold_' + str(i) + '.json')
        save_path_dev = os.path.join(GENERATED_DATASET_DIR_SEMEVAL_SPLIT_CS_KFOLD, 'CzechABSA-v2_format_SemEval2014-DEV-fold_' + str(i) + '.json')

        save_json_file(train.tolist(), save_path_train)
        save_json_file(test.tolist(), save_path_test)
        save_json_file(dev.tolist(), save_path_dev)




def save_json_file(obj2save, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(jsonpickle.encode(obj2save, indent=4))
        # json.dump(obj2save, f, indent=2)


def load_json_object(path):
    loaded_obj = None
    with open(path, 'r', encoding='utf-8') as f:
        loaded_obj = jsonpickle.decode(f.read())

    return loaded_obj

def generate_semeval_2014(args):

    # e.g., positive - price
    question_nli_b = '{} - {}'

    # e.g. price
    question_nli_m = '{}'

    if args.dataset_lang == 'cs':
        # Generate QA - M
        # aspect_examples = parse_semeval_2014(CS_SEMEVAL_2014_RAW_DATASET)
        load_train = load_json_object(CS_SEMEVAL_SPLIT_DATA_TRAIN)
        load_test = load_json_object(CS_SEMEVAL_SPLIT_DATA_TEST)
        load_dev = load_json_object(CS_SEMEVAL_SPLIT_DATA_DEV)
        data_arr = [load_train, load_test, load_dev]
        prefix_str_arr = ['train', 'test', 'dev']

        for aspect_examples, prefix in zip(data_arr, prefix_str_arr):
            generate_czech(aspect_examples, question_nli_m, question_nli_b, prefix, GENERATED_DATASET_DIR_SEMEVAL_PAIR_CS, GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA_CS)


        for i in range(CS_NUMBER_FOLDS):
            load_train = load_json_object(os.path.join(GENERATED_DATASET_DIR_SEMEVAL_SPLIT_CS_KFOLD,
                                           'CzechABSA-v2_format_SemEval2014-TRAIN-fold_' + str(i) + '.json'))
            load_test = load_json_object(os.path.join(GENERATED_DATASET_DIR_SEMEVAL_SPLIT_CS_KFOLD,
                                          'CzechABSA-v2_format_SemEval2014-TEST-fold_' + str(i) + '.json'))
            load_dev = load_json_object(os.path.join(GENERATED_DATASET_DIR_SEMEVAL_SPLIT_CS_KFOLD,
                                         'CzechABSA-v2_format_SemEval2014-DEV-fold_' + str(i) + '.json'))

            data_arr = [load_train, load_test, load_dev]
            prefix_str_arr = ['train', 'test', 'dev']

            for aspect_examples, prefix in zip(data_arr, prefix_str_arr):
                prefix = prefix + '_kfold_' + str(i)
                generate_czech(aspect_examples, question_nli_m, question_nli_b, prefix,
                               GENERATED_DATASET_DIR_SEMEVAL_PAIR_CS_KFOLD, GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA_CS_KFOLD)



    elif args.dataset_lang == 'en':
        # Test data
        # aspect_examples = parse_semeval_2014(EN_SEMEVAL_2014_RAW_DATASET_TEST)
        # # Train data
        # aspect_examples = parse_semeval_2014(EN_SEMEVAL_2014_RAW_DATASET_TRAIN)

        load_train = load_json_object(EN_SEMEVAL_SPLIT_DATA_TRAIN)
        load_test = load_json_object(EN_SEMEVAL_SPLIT_DATA_TEST)
        load_dev = load_json_object(EN_SEMEVAL_SPLIT_DATA_DEV)

        data_arr = [load_train, load_test, load_dev]
        prefix_str_arr = ['train', 'test', 'dev']

        for aspect_examples, prefix in zip(data_arr, prefix_str_arr):
            generate_english(aspect_examples, question_nli_m, question_nli_b, prefix)


def generate_english(aspect_examples, question_nli_m, question_nli_b, prefix):
    question_qa_m = 'what do you think of the {} of it ?'
    question_qa_b = 'the polarity of the aspect {} is {} .'

    # Generate QA - M
    generate_data_M(aspect_examples, GENERATED_DATASET_DIR_SEMEVAL_PAIR_EN, 'EN_' + prefix + '_QA_M',
                    SEMEVAL_2014_ASPECT_CATEGORIES, question_qa_m, EN_ASPECT_CATEGORIES_QUESTIONS_STRINGS)

    # Generate NLI - M
    generate_data_M(aspect_examples, GENERATED_DATASET_DIR_SEMEVAL_PAIR_EN, 'EN_' + prefix + '_NLI_M',
                    SEMEVAL_2014_ASPECT_CATEGORIES, question_nli_m, EN_ASPECT_CATEGORIES_QUESTIONS_STRINGS)

    # Generate QA, NLI - B
    generate_data_QA_B_NLI_B(aspect_examples, GENERATED_DATASET_DIR_SEMEVAL_PAIR_EN, 'EN_' + prefix + '_QA_B', 'EN_' + prefix + '_NLI_B',
                             SEMEVAL_2014_EN_LABELS_GENERATING, SEMEVAL_2014_ASPECT_CATEGORIES, question_qa_b,
                             question_nli_b,
                             EN_ASPECT_CATEGORIES_QUESTIONS_STRINGS, EN_SEMEVAL_2014_LABELS_GENERATING_MAPING)

    custom_qa_variant_a_question = 'Aspect {} is {} .'
    # Custom QA - variant A
    generate_custom_QA_A(aspect_examples, SEMEVAL_2014_EN_LABELS_CUSTOM_QA, SEMEVAL_2014_ASPECT_CATEGORIES,
                         'EN_' + prefix + '_CQA-A',
                         GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA_EN, custom_qa_variant_a_question,
                         EN_ASPECT_CATEGORIES_QUESTIONS_STRINGS, EN_SEMEVAL_2014_LABELS_GENERATING_MAPING)

    custom_qa_variant_b_question = 'Aspect {} is {} .'
    # Custom QA - variant B
    generate_custom_QA_B(aspect_examples, SEMEVAL_2014_EN_LABELS_CUSTOM_QA, 'EN_' + prefix + '_CQA-B',
                         GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA_EN, custom_qa_variant_b_question,
                         EN_ASPECT_CATEGORIES_QUESTIONS_STRINGS, EN_SEMEVAL_2014_LABELS_GENERATING_MAPING)

    # Custom QA - variant C
    custom_qa_variant_c_question = 'Is the aspect {} {} ?'
    custom_qa_variant_c_statement = 'Aspect {} is {} .'
    generate_custom_QA_C(aspect_examples, SEMEVAL_2014_EN_LABELS_CUSTOM_QA, SEMEVAL_2014_ASPECT_CATEGORIES,
                         'EN_' + prefix + '_CQA-C',
                         GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA_EN, custom_qa_variant_c_question,
                         custom_qa_variant_c_statement,
                         EN_ASPECT_CATEGORIES_QUESTIONS_STRINGS, EN_SEMEVAL_2014_LABELS_GENERATING_MAPING,
                         generate_all=True)

    generate_custom_QA_C(aspect_examples, SEMEVAL_2014_EN_LABELS_CUSTOM_QA, SEMEVAL_2014_ASPECT_CATEGORIES,
                         'EN_' + prefix + '_CQA-C',
                         GENERATED_DATASET_DIR_SEMEVAL_CUSTOM_QA_EN, custom_qa_variant_c_question,
                         custom_qa_variant_c_statement,
                         EN_ASPECT_CATEGORIES_QUESTIONS_STRINGS, EN_SEMEVAL_2014_LABELS_GENERATING_MAPING,
                         generate_all=False)


def generate_czech(aspect_examples, question_nli_m, question_nli_b, prefix, pair_dir, qa_dir):
    question_qa_m = 'co si myslíš o aspektu {}'
    generate_data_M(aspect_examples, pair_dir, 'CS_' + prefix + '_QA_M',
                    SEMEVAL_2014_ASPECT_CATEGORIES, question_qa_m, CS_ASPECT_CATEGORIES_QUESTIONS_STRINGS)

    # Generate NLI - M
    generate_data_M(aspect_examples, pair_dir, 'CS_' + prefix + '_NLI_M',
                    SEMEVAL_2014_ASPECT_CATEGORIES, question_nli_m, CS_ASPECT_CATEGORIES_QUESTIONS_STRINGS)

    # Generate QA, NLI - B
    question_qa_b = 'polarita aspektu {} je {} .'
    generate_data_QA_B_NLI_B(aspect_examples, pair_dir, 'CS_' + prefix + '_QA_B', 'CS_' + prefix + '_NLI_B',
                             SEMEVAL_2014_CS_LABELS_GENERATING, SEMEVAL_2014_ASPECT_CATEGORIES, question_qa_b,
                             question_nli_b,
                             CS_ASPECT_CATEGORIES_QUESTIONS_STRINGS, CS_SEMEVAL_2014_LABELS_GENERATING_MAPING)

    custom_qa_variant_a_question = 'Aspekt {} je {} .'
    # Custom QA - variant A
    generate_custom_QA_A(aspect_examples, SEMEVAL_2014_CS_LABELS_CUSTOM_QA, SEMEVAL_2014_ASPECT_CATEGORIES,
                         'CS_' + prefix + '_CQA-A',
                         qa_dir, custom_qa_variant_a_question,
                         CS_ASPECT_CATEGORIES_QUESTIONS_STRINGS, CS_SEMEVAL_2014_LABELS_GENERATING_MAPING)

    custom_qa_variant_b_question = 'Aspekt {} je {} .'
    # Custom QA - variant B
    generate_custom_QA_B(aspect_examples, SEMEVAL_2014_CS_LABELS_CUSTOM_QA, 'CS_' + prefix + '_CQA-B',
                         qa_dir, custom_qa_variant_b_question,
                         CS_ASPECT_CATEGORIES_QUESTIONS_STRINGS, CS_SEMEVAL_2014_LABELS_GENERATING_MAPING)

    # Custom QA - variant C
    custom_qa_variant_c_question = 'Is the aspect {} {} ?'
    custom_qa_variant_c_statement = 'Aspect {} is {} .'
    generate_custom_QA_C(aspect_examples, SEMEVAL_2014_CS_LABELS_CUSTOM_QA, SEMEVAL_2014_ASPECT_CATEGORIES,
                         'CS_' + prefix + '_test_CQA-C',
                         qa_dir, custom_qa_variant_c_question,
                         custom_qa_variant_c_statement,
                         CS_ASPECT_CATEGORIES_QUESTIONS_STRINGS, CS_SEMEVAL_2014_LABELS_GENERATING_MAPING,
                         generate_all=True)

    generate_custom_QA_C(aspect_examples, SEMEVAL_2014_CS_LABELS_CUSTOM_QA, SEMEVAL_2014_ASPECT_CATEGORIES,
                         'CS_' + prefix + '_test_CQA-C',
                         qa_dir, custom_qa_variant_c_question,
                         custom_qa_variant_c_statement,
                         CS_ASPECT_CATEGORIES_QUESTIONS_STRINGS, CS_SEMEVAL_2014_LABELS_GENERATING_MAPING,
                         generate_all=False)




# asi by to take slo udelat ze se to zavola dvakrat, ale je tam trochu jine poradi
def generate_data_QA_B_NLI_B(aspect_examples, output_dir, qa_file_template, nli_file_template,
                             allowed_labels, allowed_aspects, qa_template, nli_template,
                             question_aspect_mapping, question_label_mapping):
    """
    Generuje pocet aspektu krát počet labelu pro kazdej example tzn. pro semeval2014 to je 5x5 = 25 na jeden example

    Př.
    1411	0	polarita aspektu cena je pozitivní .	Jídlo taky nic moc
    1411	0	polarita aspektu cena je neutrální .	Jídlo taky nic moc
    1411	0	polarita aspektu cena je negativní .	Jídlo taky nic moc
    1411	0	polarita aspektu cena je konfliktní .	Jídlo taky nic moc
    1411	1	polarita aspektu cena je žádná .	Jídlo taky nic moc
    1411	0	polarita aspektu ostatní je pozitivní .	Jídlo taky nic moc
    1411	0	polarita aspektu ostatní je neutrální .	Jídlo taky nic moc
    1411	0	polarita aspektu ostatní je negativní .	Jídlo taky nic moc
    1411	0	polarita aspektu ostatní je konfliktní .	Jídlo taky nic moc
    1411	1	polarita aspektu ostatní je žádná .	Jídlo taky nic moc
    1411	0	polarita aspektu jídlo je pozitivní .	Jídlo taky nic moc
    1411	0	polarita aspektu jídlo je neutrální .	Jídlo taky nic moc
    1411	1	polarita aspektu jídlo je negativní .	Jídlo taky nic moc
    1411	0	polarita aspektu jídlo je konfliktní .	Jídlo taky nic moc
    1411	0	polarita aspektu jídlo je žádná .	Jídlo taky nic moc
    1411	0	polarita aspektu prostředí je pozitivní .	Jídlo taky nic moc
    1411	0	polarita aspektu prostředí je neutrální .	Jídlo taky nic moc
    1411	0	polarita aspektu prostředí je negativní .	Jídlo taky nic moc
    1411	0	polarita aspektu prostředí je konfliktní .	Jídlo taky nic moc
    1411	1	polarita aspektu prostředí je žádná .	Jídlo taky nic moc
    1411	0	polarita aspektu obsluha je pozitivní .	Jídlo taky nic moc
    1411	0	polarita aspektu obsluha je neutrální .	Jídlo taky nic moc
    1411	0	polarita aspektu obsluha je negativní .	Jídlo taky nic moc
    1411	0	polarita aspektu obsluha je konfliktní .	Jídlo taky nic moc
    1411	1	polarita aspektu obsluha je žádná .	Jídlo taky nic moc


    :param aspect_examples:
    :param output_dir:
    :param qa_file_template:
    :param nli_file_template:
    :param allowed_labels:
    :param allowed_aspects:
    :param qa_template:
    :param nli_template:
    :param question_aspect_mapping:
    :param question_label_mapping:
    :return:
    """

    logger.info("For QA B template:" + str(qa_file_template))
    output_file_qa = os.path.join(output_dir, qa_file_template + '.tsv')
    logger.info("Writing examples into file:" + output_file_qa)

    logger.info("For NLI B template:" + str(nli_file_template))
    output_file_nli = os.path.join(output_dir, nli_file_template + '.tsv')
    logger.info("Writing examples into file:" + output_file_nli)

    with open(output_file_qa, 'w', encoding='utf-8') as out_qa, \
            open(output_file_nli, 'w', encoding='utf-8') as out_nli:
        out_qa.write('id\tlabel\thelp_sentence\torig_sentence\n')
        out_nli.write('id\tlabel\thelp_sentence\torig_sentence\n')
        for i, aspect_example in enumerate(aspect_examples):
            text = aspect_example.text
            present_cat = aspect_example.present_categories
            id = aspect_example.id

            for allowed_aspect in allowed_aspects:
                for allowed_label in allowed_labels:
                    question_aspect = question_aspect_mapping[allowed_aspect]
                    question_label = question_label_mapping[allowed_label]

                    qa_final_question = qa_template.format(question_aspect, question_label)
                    nli_final_question = nli_template.format(question_label, question_aspect)

                    if allowed_aspect in present_cat:
                        category = present_cat[allowed_aspect]

                        if category.category_aspect == allowed_aspect and category.polarity == allowed_label:
                            out_qa.write(str(id) + '\t' + str(1) + '\t' + qa_final_question + '\t' + text + '\n')
                            out_nli.write(str(id) + '\t' + str(1) + '\t' + nli_final_question + '\t' + text + '\n')
                        else:
                            out_qa.write(str(id) + '\t' + str(0) + '\t' + qa_final_question + '\t' + text + '\n')
                            out_nli.write(str(id) + '\t' + str(0) + '\t' + nli_final_question + '\t' + text + '\n')
                    else:
                        if allowed_label == 'none':
                            out_qa.write(str(id) + '\t' + str(1) + '\t' + qa_final_question + '\t' + text + '\n')
                            out_nli.write(str(id) + '\t' + str(1) + '\t' + nli_final_question + '\t' + text + '\n')
                        else:
                            out_qa.write(str(id) + '\t' + str(0) + '\t' + qa_final_question + '\t' + text + '\n')
                            out_nli.write(str(id) + '\t' + str(0) + '\t' + nli_final_question + '\t' + text + '\n')


def generate_data_M(aspect_examples, output_dir, file_name_template, allowed_aspects, question, question_aspect_mapping):
    """
    Pro kazdy example generuje tolik examplu kolik je allowed_aspects, pokud ten aspekt ma label tak se mu priradi
    jinak se mu da label "none"

    Př.
    1411	none	    co si myslíš o cena	        Jídlo taky nic moc
    1411	none	    co si myslíš o ostatní	    Jídlo taky nic moc
    1411	negative	co si myslíš o jídlo	    Jídlo taky nic moc
    1411	none	    co si myslíš o prostředí	Jídlo taky nic moc
    1411	none	    co si myslíš o obsluha	    Jídlo taky nic moc

    nebo pro NLI
    1411	none	cena	Jídlo taky nic moc
    1411	none	ostatní	Jídlo taky nic moc
    1411	negative	jídlo	Jídlo taky nic moc
    1411	none	prostředí	Jídlo taky nic moc
    1411	none	obsluha	Jídlo taky nic moc

    :param aspect_examples:
    :param output_dir:
    :param file_name_template:
    :param allowed_aspects:
    :param question:
    :param question_aspect_mapping:
    :return:
    """

    logger.info("For template:" + str(file_name_template))
    output_file = os.path.join(output_dir, file_name_template + '.tsv')
    logger.info("Writing examples into file:" + output_file)

    with open(output_file, 'w', encoding='utf-8') as out:
        out.write('id\tlabel\thelp_sentence\torig_sentence\n')

        for i, aspect_example in enumerate(aspect_examples):
            text = aspect_example.text
            present_cat = aspect_example.present_categories
            id = aspect_example.id

            for allowed_aspect in allowed_aspects:

                question_aspect = question_aspect_mapping[allowed_aspect]
                final_question = question.format(question_aspect)
                if allowed_aspect in present_cat:
                    category = present_cat[allowed_aspect]
                    label = category.polarity
                    out.write(str(id) + '\t' + str(label) + '\t' + final_question + '\t' + text + '\n')
                else:
                    label = 'none'
                    out.write(str(id) + '\t' + str(label) + '\t' + final_question + '\t' + text + '\n')


def generate_custom_QA_C(aspect_examples, allowed_labels, allowed_aspects, file_template,
                         output_dir, question, statement, question_aspect_mapping, question_label_mapping,
                         generate_all=False):

    """
    if generate_all = False

    32897564#894393#2				 Is the aspect price positive ?	The bread is top notch as well. Aspect price is positive . Aspect price is neutral . Aspect price is negative . Aspect price is conflict .
    32897564#894393#2				 Is the aspect price neutral ?	The bread is top notch as well. Aspect price is positive . Aspect price is neutral . Aspect price is negative . Aspect price is conflict .
    32897564#894393#2				 Is the aspect price negative ?	The bread is top notch as well. Aspect price is positive . Aspect price is neutral . Aspect price is negative . Aspect price is conflict .
    32897564#894393#2				 Is the aspect price conflict ?	The bread is top notch as well. Aspect price is positive . Aspect price is neutral . Aspect price is negative . Aspect price is conflict .
    32897564#894393#2				 Is the aspect anecdotes positive ?	The bread is top notch as well. Aspect anecdotes is positive . Aspect anecdotes is neutral . Aspect anecdotes is negative . Aspect anecdotes is conflict .
    32897564#894393#2				 Is the aspect anecdotes neutral ?	The bread is top notch as well. Aspect anecdotes is positive . Aspect anecdotes is neutral . Aspect anecdotes is negative . Aspect anecdotes is conflict .
    32897564#894393#2				 Is the aspect anecdotes negative ?	The bread is top notch as well. Aspect anecdotes is positive . Aspect anecdotes is neutral . Aspect anecdotes is negative . Aspect anecdotes is conflict .
    32897564#894393#2				 Is the aspect anecdotes conflict ?	The bread is top notch as well. Aspect anecdotes is positive . Aspect anecdotes is neutral . Aspect anecdotes is negative . Aspect anecdotes is conflict .
    32897564#894393#2	32	57	 Aspect food is positive .	 Is the aspect food positive ?	The bread is top notch as well. Aspect food is positive . Aspect food is neutral . Aspect food is negative . Aspect food is conflict .
    32897564#894393#2				 Is the aspect food neutral ?	The bread is top notch as well. Aspect food is positive . Aspect food is neutral . Aspect food is negative . Aspect food is conflict .
    32897564#894393#2				 Is the aspect food negative ?	The bread is top notch as well. Aspect food is positive . Aspect food is neutral . Aspect food is negative . Aspect food is conflict .
    32897564#894393#2				 Is the aspect food conflict ?	The bread is top notch as well. Aspect food is positive . Aspect food is neutral . Aspect food is negative . Aspect food is conflict .
    32897564#894393#2				 Is the aspect ambience positive ?	The bread is top notch as well. Aspect ambience is positive . Aspect ambience is neutral . Aspect ambience is negative . Aspect ambience is conflict .
    32897564#894393#2				 Is the aspect ambience neutral ?	The bread is top notch as well. Aspect ambience is positive . Aspect ambience is neutral . Aspect ambience is negative . Aspect ambience is conflict .
    32897564#894393#2				 Is the aspect ambience negative ?	The bread is top notch as well. Aspect ambience is positive . Aspect ambience is neutral . Aspect ambience is negative . Aspect ambience is conflict .
    32897564#894393#2				 Is the aspect ambience conflict ?	The bread is top notch as well. Aspect ambience is positive . Aspect ambience is neutral . Aspect ambience is negative . Aspect ambience is conflict .
    32897564#894393#2				 Is the aspect service positive ?	The bread is top notch as well. Aspect service is positive . Aspect service is neutral . Aspect service is negative . Aspect service is conflict .
    32897564#894393#2				 Is the aspect service neutral ?	The bread is top notch as well. Aspect service is positive . Aspect service is neutral . Aspect service is negative . Aspect service is conflict .
    32897564#894393#2				 Is the aspect service negative ?	The bread is top notch as well. Aspect service is positive . Aspect service is neutral . Aspect service is negative . Aspect service is conflict .
    32897564#894393#2

    if generate_all=True

    32897564#894393#2				 Is the aspect price positive ?	The bread is top notch as well. Aspect price is positive . Aspect price is neutral . Aspect price is negative . Aspect price is conflict .
    32897564#894393#2				 Is the aspect price neutral ?	The bread is top notch as well. Aspect price is positive . Aspect price is neutral . Aspect price is negative . Aspect price is conflict .
    32897564#894393#2				 Is the aspect price negative ?	The bread is top notch as well. Aspect price is positive . Aspect price is neutral . Aspect price is negative . Aspect price is conflict .
    32897564#894393#2				 Is the aspect price conflict ?	The bread is top notch as well. Aspect price is positive . Aspect price is neutral . Aspect price is negative . Aspect price is conflict .
    32897564#894393#2				 Is the aspect anecdotes positive ?	The bread is top notch as well. Aspect anecdotes is positive . Aspect anecdotes is neutral . Aspect anecdotes is negative . Aspect anecdotes is conflict .
    32897564#894393#2				 Is the aspect anecdotes neutral ?	The bread is top notch as well. Aspect anecdotes is positive . Aspect anecdotes is neutral . Aspect anecdotes is negative . Aspect anecdotes is conflict .
    32897564#894393#2				 Is the aspect anecdotes negative ?	The bread is top notch as well. Aspect anecdotes is positive . Aspect anecdotes is neutral . Aspect anecdotes is negative . Aspect anecdotes is conflict .
    32897564#894393#2				 Is the aspect anecdotes conflict ?	The bread is top notch as well. Aspect anecdotes is positive . Aspect anecdotes is neutral . Aspect anecdotes is negative . Aspect anecdotes is conflict .
    32897564#894393#2	32	57	 Aspect food is positive .	 Is the aspect food positive ?	The bread is top notch as well. Aspect food is positive . Aspect food is neutral . Aspect food is negative . Aspect food is conflict .
    32897564#894393#2	32	57	 Aspect food is positive .	 Is the aspect food neutral ?	The bread is top notch as well. Aspect food is positive . Aspect food is neutral . Aspect food is negative . Aspect food is conflict .
    32897564#894393#2	32	57	 Aspect food is positive .	 Is the aspect food negative ?	The bread is top notch as well. Aspect food is positive . Aspect food is neutral . Aspect food is negative . Aspect food is conflict .
    32897564#894393#2	32	57	 Aspect food is positive .	 Is the aspect food conflict ?	The bread is top notch as well. Aspect food is positive . Aspect food is neutral . Aspect food is negative . Aspect food is conflict .
    32897564#894393#2				 Is the aspect ambience positive ?	The bread is top notch as well. Aspect ambience is positive . Aspect ambience is neutral . Aspect ambience is negative . Aspect ambience is conflict .
    32897564#894393#2				 Is the aspect ambience neutral ?	The bread is top notch as well. Aspect ambience is positive . Aspect ambience is neutral . Aspect ambience is negative . Aspect ambience is conflict .
    32897564#894393#2				 Is the aspect ambience negative ?	The bread is top notch as well. Aspect ambience is positive . Aspect ambience is neutral . Aspect ambience is negative . Aspect ambience is conflict .
    32897564#894393#2				 Is the aspect ambience conflict ?	The bread is top notch as well. Aspect ambience is positive . Aspect ambience is neutral . Aspect ambience is negative . Aspect ambience is conflict .
    32897564#894393#2				 Is the aspect service positive ?	The bread is top notch as well. Aspect service is positive . Aspect service is neutral . Aspect service is negative . Aspect service is conflict .
    32897564#894393#2				 Is the aspect service neutral ?	The bread is top notch as well. Aspect service is positive . Aspect service is neutral . Aspect service is negative . Aspect service is conflict .
    32897564#894393#2				 Is the aspect service negative ?	The bread is top notch as well. Aspect service is positive . Aspect service is neutral . Aspect service is negative . Aspect service is conflict .
    32897564#894393#2				 Is the aspect service conflict ?	The bread is top notch as well. Aspect service is positive . Aspect service is neutral . Aspect service is negative . Aspect service is conflict .


    :param aspect_examples:
    :param allowed_labels:
    :param allowed_aspects:
    :param file_template:
    :param output_dir:
    :param question:
    :param statement:
    :param question_aspect_mapping:
    :param question_label_mapping:
    :param generate_all:
    :return:
    """

    file_template += '-all_' + str(generate_all)
    logger.info("Generating output for custom QA variant C")
    output_file_c = os.path.join(output_dir, file_template + '-variant-C.tsv')
    logger.info("Writing examples into file:" + output_file_c)

    with open(output_file_c, 'w', encoding ='utf-8') as out_c:
        out_c.write('id\tspan_start\tspan_end\tspan_text\tsentence_question\tsentence_example\n')

        for i, aspect_example in enumerate(aspect_examples):
            text = aspect_example.text.strip()
            present_cat = aspect_example.present_categories
            id = aspect_example.id

            for allowed_aspect in allowed_aspects:
                for allowed_label in allowed_labels:
                    question_aspect = question_aspect_mapping[allowed_aspect]
                    question_label = question_label_mapping[allowed_label]
                    final_question = ' ' + question.format(question_aspect, question_label)

                    output_text = text
                    span_start = None
                    span_end = None
                    span_text = None
                    is_labeled = False
                    for allowed_label_tmp in allowed_labels:
                        statement_label = question_label_mapping[allowed_label_tmp]
                        statement_text = ' ' + statement.format(question_aspect, statement_label)

                        tmp_start = len(output_text) + 1  # bcs there is space

                        output_text = output_text + statement_text

                        if allowed_aspect in present_cat:
                            category = present_cat[allowed_aspect]

                            if generate_all:
                                condition = category.polarity == allowed_label_tmp
                            else:
                                condition = (category.polarity == allowed_label_tmp and category.polarity == allowed_label)

                            if condition:
                                if span_start != None:
                                    raise ValueError("The span_start has been already assigned")
                                span_start = tmp_start
                                span_end = len(output_text)
                                span_text = statement_text
                                is_labeled = True

                    if is_labeled == True:
                        out_c.write(str(id) + '\t' + str(span_start) + '\t' + str(span_end) + '\t' +
                                  str(span_text) + '\t' + str(final_question) + '\t' + output_text + '\n')
                    else:
                        out_c.write(str(id) + '\t' + str('') + '\t' + str('') + '\t' +
                                    str('') + '\t' + str(final_question) + '\t' + output_text + '\n')





def generate_custom_QA_B(aspect_examples, allowed_labels, file_template,
                         output_dir, question, question_aspect_mapping, question_label_mapping):

    """
    32894246#870052#0	186	211	 Aspect food is conflict .	Certainly not the best sushi in New York, however, it is always fresh, and the place is very clean, sterile. Aspect food is positive . Aspect food is neutral . Aspect food is negative . Aspect food is conflict .
    32894246#870052#0	109	138	 Aspect ambience is positive .	Certainly not the best sushi in New York, however, it is always fresh, and the place is very clean, sterile. Aspect ambience is positive . Aspect ambience is neutral . Aspect ambience is negative . Aspect ambience is conflict .
    32894246#870052#1	54	84	 Aspect anecdotes is positive .	I trust the people at Go Sushi, it never disappoints. Aspect anecdotes is positive . Aspect anecdotes is neutral . Aspect anecdotes is negative . Aspect anecdotes is conflict .


    :param aspect_examples:
    :param allowed_labels:
    :param file_template:
    :param output_dir:
    :param question:
    :param question_aspect_mapping:
    :param question_label_mapping:
    :return:
    """

    logger.info("Generating output for custom QA variant B")
    output_file = os.path.join(output_dir, file_template + '-variant-B.tsv')
    logger.info("Writing examples into file:" + output_file)

    with open(output_file, 'w', encoding='utf-8') as out:
        out.write('id\tspan_start\tspan_end\tspan_text\tsentence_example\n')

        for i, aspect_example in enumerate(aspect_examples):
            text = aspect_example.text.strip()
            id = aspect_example.id

            for aspect_category in aspect_example.aspect_categories:
                output_text = text
                span_start = None
                for allowed_label in allowed_labels:
                    question_aspect = question_aspect_mapping[aspect_category.category_aspect]
                    question_label = question_label_mapping[allowed_label]
                    final_question = ' ' + question.format(question_aspect, question_label)

                    tmp_start = len(output_text) + 1 # bcs there is space

                    output_text = output_text + final_question

                    if aspect_category.polarity == allowed_label:
                        if span_start != None:
                            raise ValueError("The span_start has been already assigned")
                        span_start = tmp_start
                        span_end = len(output_text)
                        span_text = final_question

                out.write(str(id) + '\t' + str(span_start) + '\t' + str(span_end) + '\t' + str(
                    span_text) + '\t' + output_text + '\n')






def generate_custom_QA_A(aspect_examples, allowed_labels, allowed_aspects, file_template,
                         output_dir, question, question_aspect_mapping, question_label_mapping):
    """
    Produkuje examply, pocet aspektu x pocet labelu, zde je to 5x4, 4 jsou labely - conflict, positive, negative, neutral

    vystup
    id span_start span_end span_text sentence_example

    32897564#894393#2				The bread is top notch as well. Aspect price is positive .
    32897564#894393#2				The bread is top notch as well. Aspect price is neutral .
    32897564#894393#2				The bread is top notch as well. Aspect price is negative .
    32897564#894393#2				The bread is top notch as well. Aspect price is conflict .
    32897564#894393#2				The bread is top notch as well. Aspect anecdotes is positive .
    32897564#894393#2				The bread is top notch as well. Aspect anecdotes is neutral .
    32897564#894393#2				The bread is top notch as well. Aspect anecdotes is negative .
    32897564#894393#2				The bread is top notch as well. Aspect anecdotes is conflict .
    32897564#894393#2	32	56	 Aspect food is positive .	The bread is top notch as well. Aspect food is positive .
    32897564#894393#2				The bread is top notch as well. Aspect food is neutral .
    32897564#894393#2				The bread is top notch as well. Aspect food is negative .
    32897564#894393#2				The bread is top notch as well. Aspect food is conflict .
    32897564#894393#2				The bread is top notch as well. Aspect ambience is positive .
    32897564#894393#2				The bread is top notch as well. Aspect ambience is neutral .
    32897564#894393#2				The bread is top notch as well. Aspect ambience is negative .
    32897564#894393#2				The bread is top notch as well. Aspect ambience is conflict .
    32897564#894393#2				The bread is top notch as well. Aspect service is positive .
    32897564#894393#2				The bread is top notch as well. Aspect service is neutral .
    32897564#894393#2				The bread is top notch as well. Aspect service is negative .
    32897564#894393#2				The bread is top notch as well. Aspect service is conflict .


    :param aspect_examples:
    :return:
    """

    logger.info("Generating output for custom QA variant A")
    output_file = os.path.join(output_dir, file_template + '-variant-A.tsv')
    logger.info("Writing examples into file:" + output_file)

    with open(output_file, 'w', encoding='utf-8') as out:
        out.write('id\tspan_start\tspan_end\tspan_text\tsentence_example\n')

        for i, aspect_example in enumerate(aspect_examples):
            text = aspect_example.text
            text = text.strip()
            present_aspect_cat = aspect_example.present_categories
            id = aspect_example.id

            for allowed_aspect in allowed_aspects:
                for allowed_label in allowed_labels:
                    question_aspect = question_aspect_mapping[allowed_aspect]
                    question_label = question_label_mapping[allowed_label]

                    final_question = ' ' + question.format(question_aspect, question_label)
                    output_text = text +  final_question
                    span_start = len(text) + 1 # bcs there is space
                    span_end = len(output_text)
                    span_text = final_question

                    written = False
                    if allowed_aspect in present_aspect_cat:
                        category = present_aspect_cat[allowed_aspect]
                        if category.category_aspect == allowed_aspect and category.polarity == allowed_label:
                            out.write(str(id)+'\t'+str(span_start)+'\t'+str(span_end)+'\t'+str(span_text)+'\t'+output_text+'\n')
                            written = True

                    if written == False:
                        out.write(str(id) + '\t' + '\t' + '\t' + '\t' + output_text + '\n')





if __name__ == '__main__':
    sys.argv.extend(['--split_data'])

    # sys.argv.extend(['--dataset_lang', 'en'])
    # sys.argv.extend(['--source', 'semeval2014'])

    sys.argv.extend(['--dataset_lang', 'cs'])
    sys.argv.extend(['--source', 'semeval2014'])

    main()
