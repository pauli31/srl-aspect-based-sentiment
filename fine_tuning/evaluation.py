
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import logging
import numpy as np
import wandb

from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logging.root.setLevel(level=logging.INFO)
logger = logging.getLogger(__name__)

# Evaluation partly based on project from ABSA-BERT-pair-master https://github.com/HSLCY/ABSA-BERT-pair
#


def evaluate_predictions(y_pred, y_test, average='macro'):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=average)
    precision = precision_score(y_test, y_pred, average=average)
    recall = recall_score(y_test, y_pred, average=average)

    return f1, accuracy, precision, recall


def prepare_labels4aspect_cat_eval(y_pred, y_true, y_pred_probs, solution_type, dataset_loader):
    num_sentiment_classes = dataset_loader.get_sentiment_classes_num()
    num_aspect_classes = dataset_loader.get_aspect_classes_num()
    generated_sentiment_labels_map = dataset_loader.get_generated_sentiment_labels_mapping()

    if solution_type == 'QA_B' or solution_type == 'NLI_B':

        # examples per sentences
        examples_per_sent = num_sentiment_classes * num_aspect_classes

        y_pred_converted = []
        y_pred_converted_probs = []
        y_true_converted = []

        # iterate over each example
        for i in range(len(y_pred)//examples_per_sent):

            # iterate over each aspect
            for j in range(num_aspect_classes):

                probs_positive = []
                y_true_values = []
                # iterate over each sentiment
                for k in range(num_sentiment_classes):
                    index_in_example = i * examples_per_sent + j * num_sentiment_classes + k
                    true_val = y_true[index_in_example]
                    pred_prob = y_pred_probs[index_in_example]
                    # score for positive class
                    probs_positive.append(float(pred_prob[1]))
                    y_true_values.append(true_val)

                    # if we are at the last example for a given aspect j
                    # the last one is the none sentiment label
                    # and here we find the best prediction for a given aspect
                    if (k+1) == num_sentiment_classes:
                        # the order is different in generated texts,
                        # bcs there is differnt order of the sentiment labels
                        y_true_conv = np.argmax(y_true_values)
                        y_true_class_name = generated_sentiment_labels_map[y_true_conv]
                        y_true_conv = dataset_loader.get_label4text_evaluation(y_true_class_name)

                        # here we have to select the index of the highest probabilites
                        # bcs there could be more than one positive prediction
                        y_pred_conv = np.argmax(probs_positive)
                        y_pred_class_name = generated_sentiment_labels_map[y_pred_conv]
                        y_pred_conv = dataset_loader.get_label4text_evaluation(y_pred_class_name)

                        # now we want positive probabilites for each sentiment prediction
                        # but we have to put them on right indices according to the map
                        # bcs these probs are according to the generated order which is different
                        # each index denotes the given class
                        y_pred_conv_prob = np.zeros(num_sentiment_classes)
                        for m in range(num_sentiment_classes):
                            prob = probs_positive[m]
                            tmp_class_name = generated_sentiment_labels_map[m]
                            new_index =dataset_loader.get_label4text_evaluation(tmp_class_name)
                            y_pred_conv_prob[new_index] = prob

                        y_true_converted.append(y_true_conv)
                        y_pred_converted.append(y_pred_conv)
                        y_pred_converted_probs.append(y_pred_conv_prob)


        print("Data from Binary task converted")
    elif solution_type == 'NLI_M' or solution_type == 'QA_M':
        y_pred_converted = y_pred
        y_true_converted = y_true
        y_pred_converted_probs = y_pred_probs.detach().numpy()
    else:
        raise Exception("Unkown type:" + str(solution_type))

    return y_true_converted, y_pred_converted, y_pred_converted_probs


def compute_semeval_acc(y_true, y_pred, y_pred_probs, classes_to_eval, none_class, classes_to_eval_names):
    # it expects that only positions in y_pred_probs for y_pred_probs are going to be greater than zero,
    # other classses that are not

    assert len(y_pred) == len(y_true)
    assert len(y_true) == len(y_pred_probs)

    total = 0
    total_right = 0
    new_y_pred = []
    new_y_true = []

    for i in range(len(y_true)):

        # now we have to set all probs that are not in classes that are going to be evaluated to zero
        tmp_probs = y_pred_probs[i]
        for class_index in range(len(tmp_probs)):
            if class_index not in classes_to_eval or class_index == none_class:
                tmp_probs[class_index] = 0.0

        # ignore the none label or the other labels
        if y_true[i] == none_class or y_true[i] not in classes_to_eval: continue

        total +=1
        tmp = y_pred[i]

        # if we predict none class, or any other class that is not in the classes_to_eval we select the second highest
        if tmp == none_class or tmp not in classes_to_eval:
            tmp = np.argmax(tmp_probs)

        # create new valid pred and y_true
        new_y_pred.append(tmp)
        new_y_true.append(y_true[i])

        if y_true[i] == tmp:
            total_right +=1

    if total == 0:
        sentiment_acc = 0.0
    else:
        sentiment_acc = total_right / total

    f1, accuracy, precision, recall = evaluate_predictions(new_y_pred, new_y_true)
    if sentiment_acc != accuracy:
        logger.info("The accuracies are different:" + str(accuracy) +  " sentiment acc:" + str(sentiment_acc))

    try:
        class_report = classification_report(new_y_true, new_y_pred, labels=classes_to_eval, target_names=classes_to_eval_names)
    except Exception as e:
        logger.error(e)
        logger.info("Cannot compute class report")
        class_report = "NONE_CLASS_REPORT"

    return f1, sentiment_acc, precision, recall, class_report


def compute_aspect_cat_semeval_PRF(y_true, y_pred, allowed_sentiment_clases, ignore_labels, num_aspect_classes):
    '''

    :param y_true:
    :param y_pred:
    :param allowed_sentiment_clases:
    :param ignore_labels: list of numbers that refers to labels that will be skipped, typicaly it is for example
                          for the label none that has label equal 4 in English and  3 in Czech
    :return:
    '''

    """
        Based on 
        Calculates "Micro P R F" of aspect detection task of SemEval-2014.
    """

    assert len(y_pred) == len(y_true)

    # pocet vsech aspektu v gold datech
    gold_size_all = 0
    # pocet predikovanych aspektu
    pred_size_all = 0
    # pocet shodujicich se aspektu
    pred_gold_size_all = 0

    # we iterate over all predictions for each example
    # there is class_num aspects and we predict each aspect whether it is there or not
    # proste vezmeme predikce a iterujeme po n-ticich
    for i in range(len(y_pred)//num_aspect_classes):

        # set predikovaných aspektů
        pred_s = set()
        # set gold aspektů
        gold_s = set()

        # iterate over all five aspects
        for j in range(num_aspect_classes):

            # if prediction is not in the ignore labels we count it
            if y_pred[i*num_aspect_classes+j] not in ignore_labels:
                if y_pred[i*num_aspect_classes+j] in allowed_sentiment_clases:
                    pred_s.add(j)
                else:
                    raise Exception("There is predicted class that is not allowed:" + str(y_pred[i*num_aspect_classes+j]))

            #  if gold label is not in the ignore labels
            if y_true[i*num_aspect_classes+j] not in ignore_labels:
                if y_true[i*num_aspect_classes+j] in allowed_sentiment_clases:
                    gold_s.add(j)
                else:
                    raise Exception("There is gold class that is not allowed:" + str(y_true[i*num_aspect_classes+j]))

        # This was in the original script but this is not correct, since
        # If the model predicts some aspect and there are no aspects in the gold data
        # it makes better precision
        # if len(gold_s) == 0: continue

        pred_gold = pred_s.intersection(gold_s)
        pred_gold_size_all += len(pred_gold)
        gold_size_all += len(gold_s)
        pred_size_all += len(pred_s)

    logger.info("allowed_sentiment_clases:" + str(allowed_sentiment_clases))
    logger.info("num_aspect_classes:" + str(num_aspect_classes))
    logger.info("ignore_labels:" + str(ignore_labels))
    logger.info("y_pred:" + str(y_pred))
    logger.info("y_true:" + str(y_true))
    logger.info("pred_gold_size_all:" + str(pred_gold_size_all))
    logger.info("gold_size_all:" + str(gold_size_all))
    logger.info("pred_size_all:" + str(pred_size_all))

    try:
        prec = pred_gold_size_all / pred_size_all
    except Exception as e:
        logger.error(e)
        logger.info("Failed to compute Prec for aspect")
        prec = 0

    try:
        rec = pred_gold_size_all / gold_size_all
    except Exception as e:
        logger.info("Failed to compute Rec for aspect")
        rec = 0

    try:
        rec = pred_gold_size_all / gold_size_all
        fmicro = 2 * prec * rec / (prec + rec)
    except Exception as e:
        logger.error(e)
        logger.info("Failed to compute f-micro for aspect")
        fmicro = 0


    return fmicro, prec, rec


class AspectEvalResult(object):
    def __init__(self, f1_micro, prec, rec):
        self.f1_micro = f1_micro
        self.prec = prec
        self.rec = rec


class GeneralResult(object):
    def __init__(self, f1_macro, acc, prec, rec, class_report):
        self.f1_macro = f1_macro
        self.acc = acc
        self.prec = prec
        self.rec = rec
        self.class_report = class_report


class CategoryEvaluationResult(object):
    def __init__(self, full_res, aspect_res, sentiment_four, sentiment_three, sentiment_binary):
        self.full_res = full_res
        self.aspect_res = aspect_res
        self.sentiment_four = sentiment_four
        self.sentiment_three = sentiment_three
        self.sentiment_binary = sentiment_binary


def print_category_results(data_type, result, use_wandb, args):

    def print_general(data_type, desc, general_res):
        desc = data_type + "_" + desc
        print(70 * '*-')
        print(desc + " report")
        print(general_res.class_report)
        print(desc + " F1 macro:{:.4f}".format(general_res.f1_macro))
        print(desc + " accuracy:{:.4f}".format(general_res.acc))
        print(desc + " precision:{:.4f}".format(general_res.prec))
        print(desc + " recall:{:.4f}".format(general_res.rec))
        print(70 * '*-')

        if use_wandb is True:
            try:
                wandb.run.summary['f1_macro_' + desc] = general_res.f1_macro
                wandb.run.summary['accuracy_' + desc] = general_res.acc
                wandb.run.summary['precision_' + desc] = general_res.prec
                wandb.run.summary['recall_' + desc] = general_res.rec
            except Exception as e:
                logger.error("Error WANDB with exception e:" + str(e))

        results_head = '\tF1 Macro_' + str(desc) + '\t' + 'Accuracy_' + str(desc) + '\t' + 'Precision_' + str(desc) + '\t' + 'Recall_' + str(desc)
        results = f'{general_res.f1_macro}\t{general_res.acc}\t{general_res.prec}\t{general_res.rec}\t\t\t'

        return results_head, results



    def print_aspect(data_type, desc, aspect_res):
        desc = data_type + "_" + desc
        print(70 * '*-')
        print(desc + " F1 micro :{:.4f}".format(aspect_res.f1_micro))
        print(desc + " precision:{:.4f}".format(aspect_res.prec))
        print(desc + " recall:{:.4f}".format(aspect_res.rec))
        print(70 * '*-')

        if use_wandb is True:
            try:
                wandb.run.summary['f1_micro_aspect_' + desc] = aspect_res.f1_micro
                wandb.run.summary['precision_aspect_' + desc] = aspect_res.prec
                wandb.run.summary['recall_aspect_' + desc] = aspect_res.rec
            except Exception as e:
                logger.error("Error WANDB with exception e:" + str(e))

        results_head = '\tF1 Micro_' + str(desc) + '\t' + 'Precision_' + str(desc) + '\t' + 'Recall_' + str(desc) + '\t'
        results = f'\t{aspect_res.f1_micro}\t{aspect_res.prec}\t{aspect_res.rec}\t\t\t'

        return results_head, results

    results_head = ''
    results = ''

    head, res = print_aspect(data_type, "Aspect", result.aspect_res)
    results_head += head
    results += res

    head, res = print_general(data_type, "Full", result.full_res)
    results_head += head
    results += res

    head, res = print_general(data_type, "Four_Sentiment", result.sentiment_four)
    results_head += head
    results += res

    head, res = print_general(data_type, "Three_Sentiment", result.sentiment_three)
    results_head += head
    results += res

    head, res = print_general(data_type, "Binary_Sentiment", result.sentiment_binary)
    results_head += head
    results += res

    return results_head, results