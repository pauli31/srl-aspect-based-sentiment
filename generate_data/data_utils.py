import logging
from xml.dom import minidom
import sys
import os

# https://stackoverflow.com/questions/43728431/relative-imports-modulenotfounderror-no-module-named-x
# need to set because the config is in the parent directory
import jsonpickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT, CS_SEMEVAL_SPLIT_DATA_TRAIN, CS_SEMEVAL_SPLIT_DATA_TEST, \
    CS_SEMEVAL_SPLIT_DATA_DEV
from domain.aspect_example import AspectTerm, AspectCategory, AspectExample

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)

def parse_semeval_2014(file_path_in):
    xmldoc = minidom.parse(file_path_in)
    sentences = xmldoc.getElementsByTagName('sentence')
    logger.info("Total sentences:" + str(len(sentences)))

    examples = []
    for i, sentence in enumerate(sentences):
        sent_id = sentence.attributes['id'].value
        # logger.info("Processing sentence with id:" + str(sent_id))

        texts_elements = sentence.getElementsByTagName('text')
        if len(texts_elements) != 1:
            raise Exception("There not one text for this sentence:" + str(len(texts_elements)))
        text = sentence.getElementsByTagName('text')[0].childNodes[0].nodeValue

        terms = []
        aspect_terms = sentence.getElementsByTagName('aspectTerm')
        for j, aspect_term_node in enumerate(aspect_terms):
            term = aspect_term_node.attributes['term'].value
            polarity = aspect_term_node.attributes['polarity'].value
            from_pos = aspect_term_node.attributes['from'].value
            to_pos = aspect_term_node.attributes['to'].value
            aspect_term = AspectTerm(term, polarity, from_pos, to_pos)
            terms.append(aspect_term)

        categories = []
        set_cat = set()
        aspect_categories = sentence.getElementsByTagName('aspectCategory')
        for j, aspect_category_node in enumerate(aspect_categories):
            category = aspect_category_node.attributes['category'].value
            polarity = aspect_category_node.attributes['polarity'].value
            aspect_category = AspectCategory(category, polarity)
            categories.append(aspect_category)
            if category in set_cat:
                print("")
            else:
                set_cat.add(category)

        aspect_example = AspectExample(sent_id, text, terms, categories)
        examples.append(aspect_example)
    return examples

def load_json_object(path):
    loaded_obj = None
    with open(path, 'r', encoding='utf-8') as f:
        loaded_obj = jsonpickle.decode(f.read())

    return loaded_obj

if __name__ == '__main__':
    data = parse_semeval_2014("/data/ondfa@gapps.zcu.cz/IdeaProjects/aspect-based-sentiment/data/raw-datasets/czech/CzechABSA-v2_format_SemEval2014.xml")
    text = "\n".join([sent.text for sent in data])
    with open("absaData.txt", "w") as f:
        f.write(text)

    load_train = load_json_object(CS_SEMEVAL_SPLIT_DATA_TRAIN)
    text = "\n".join([sent.text for sent in load_train])
    with open("absaData_train.txt", "w") as f:
        f.write(text)
    load_test = load_json_object(CS_SEMEVAL_SPLIT_DATA_TEST)
    text = "\n".join([sent.text for sent in load_test])
    with open("absaData_test.txt", "w") as f:
        f.write(text)
    load_dev = load_json_object(CS_SEMEVAL_SPLIT_DATA_DEV)
    text = "\n".join([sent.text for sent in load_dev])
    with open("absaData_dev.txt", "w") as f:
        f.write(text)