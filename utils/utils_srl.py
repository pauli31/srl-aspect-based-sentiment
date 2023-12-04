import logging
import math
import os
import random
import subprocess
import tempfile

import wandb
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union, Tuple
import numpy as np
import torch
import itertools

from collections import defaultdict

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available

from config import SRL_DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for token classification.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    pred: str
    labels: Optional[List[str]]


@dataclass
class InputExampleE2E:
    guid: str
    words: List[str]
    pred: List[int]
    pred_lemma: List[str]
    labels: Optional[List[List[str]]]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    position_ids: Optional[List[int]] = None

    def __repr__(self):
        return f"input ids: {str(self.input_ids)}\n attention mask: {str(self.attention_mask)}\n label ids:{str(self.label_ids)}\n position ids: {str(self.position_ids)}\n\n"

    def __str__(self):
        return self.__repr__()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.txt")
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        pred = ""
        for line in f:
            if line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, pred=pred, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
                    pred = ""
            else:
                splits = line.split("\t")
                if len(splits) > 1:
                    words.append(splits[0])
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    pred = line.replace("\n", "").split("\t")[0]
        if words:
            examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, pred=pred, labels=labels))
    return examples


def read_examples_from_file_e2e(data_dir, mode: Union[Split, str]) -> List[InputExampleE2E]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.txt")
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        preds = []
        last_roles = None
        roles_count = 0
        for line in f.read().splitlines():
            if line == "" or line == "\n":
                if words:
                    examples.append(
                        InputExampleE2E(guid=f"{mode}-{guid_index}", words=words, pred=preds, labels=labels, pred_lemma=[]))
                    guid_index += 1
                    words = []
                    labels = []
                    preds = []
                    last_roles = None
            else:
                splits = line.split("\t")
                words.append(splits[1])
                ls = ["O" if label is "_" or label is "" else label for label in splits[-1].split(";")[2:]]
                for i in range(len(ls)):
                    if ls[i] is not "O":
                        ls[i] = "B-" + ls[i] if last_roles is None or last_roles[i] != ls[i] else "I-" + ls[i]
                        # if "B-" in ls[i]:
                        #     roles_count += 1
                labels.append(ls)
                last_roles = ["O" if label is "_" else label for label in splits[-1].split(";")[2:]]
                if splits[-1].split(";")[0] is not "_":
                    preds.append(int(splits[0]))
        if words:
            examples.append(InputExampleE2E(guid=f"{mode}-{guid_index}", words=words, pred=preds, labels=labels, pred_lemma=[]))
    # print(roles_count)
    return examples

def read_examples_from_ontonotes(data_dir, mode: Union[Split, str]) -> List[InputExampleE2E]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.txt")
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        preds = []
        pred_lemmas = []
        last_roles = None
        roles_count = 0
        for line in f.read().splitlines():
            if line == "" or line == "\n":
                if words:
                    examples.append(
                        InputExampleE2E(guid=f"{mode}-{guid_index}", words=words, pred=preds, labels=labels, pred_lemma=pred_lemmas))
                    guid_index += 1
                    words = []
                    labels = []
                    preds = []
                    pred_lemmas = []
                    last_roles = None
            else:
                splits = line.split()
                if len(splits) < 10:
                    continue
                words.append(splits[3])
                ls = splits[11:-1]
                if last_roles is None:
                    last_roles = ["O"] * len(ls)
                for i in range(len(ls)):
                    if ls[i][0] == "(":
                        if last_roles[i] != "O":
                            roles_count += 1
                        # TODO model overlapping arguments
                        role = ls[i].replace(")", "").replace("*", "").split("(")[-1]
                        role = role.replace("-", "_")
                        last_roles[i] = "I-" + role
                        role = "B-" + role
                    else:
                        role = last_roles[i]
                    if ls[i][-1] == ")":
                        last_roles[i] = "O"
                    ls[i] = role
                labels.append(ls)
                if splits[7] != "-":
                    preds.append(int(splits[2]) + 1)
                    pred_lemmas.append(splits[6])
        if words:
            examples.append(InputExampleE2E(guid=f"{mode}-{guid_index}", words=words, pred=preds, labels=labels, pred_lemma=pred_lemmas))
    print(roles_count)
    return examples

def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        sequence_b_segment_id=1,
        mask_padding_with_zero=True,
        gold_predicates=True,
        pred_sampling_prob=1.0
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers
    # TODO model argument parts separately
    split = examples[0].guid[:examples[0].guid.index("-")] if len(examples) > 0 else ""
    label_map = defaultdict(lambda: label_list.index("O"))
    for i, label in enumerate(label_list):
        label_map[label] = i
    # label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = [[] for _ in example.pred]
        index = 1
        preds_tokens = []
        pred_starts = []
        dummy_pred_tokens = []
        dummy_pred_starts = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                for k, pred in enumerate(example.pred):
                    pred_start = len(tokens)
                    if cls_token is not None and not cls_token_at_end:
                        pred_start += 1
                    if index == int(pred):
                        preds_tokens.append(word_tokens)
                        pred_starts.append(pred_start)
                    elif not gold_predicates and (split != "train" or random.random() < pred_sampling_prob):
                        dummy_pred_tokens.append(word_tokens)
                        dummy_pred_starts.append(pred_start)
                    label_ids[k].extend([label_map[label[k]]] + [pad_token_label_id] * (len(word_tokens) - 1))
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens

                index += 1

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        for pred_id in range(len(example.pred)):
            tmp_tokens = tokens[:]
            features.append(generate_data_for_pred(cls_token, cls_token_at_end, cls_token_segment_id, tmp_tokens,
                                                   label_ids[pred_id], mask_padding_with_zero, max_seq_length, pad_on_left,
                                                   pad_token,
                                                   pad_token_label_id, pad_token_segment_id, pred_starts[pred_id],
                                                   preds_tokens[pred_id],
                                                   sep_token, sep_token_extra, sequence_a_segment_id,
                                                   sequence_b_segment_id, tokenizer, ex_index))
        for pred_id in range(len(dummy_pred_starts)):
            tmp_tokens = tokens[:]
            tmp_label_ids = [pad_token_label_id] * len(tokens)
            features.append(generate_data_for_pred(cls_token, cls_token_at_end, cls_token_segment_id, tmp_tokens,
                                                   tmp_label_ids, mask_padding_with_zero, max_seq_length, pad_on_left,
                                                   pad_token,
                                                   pad_token_label_id, pad_token_segment_id, dummy_pred_starts[pred_id],
                                                   dummy_pred_tokens[pred_id],
                                                   sep_token, sep_token_extra, sequence_a_segment_id,
                                                   sequence_b_segment_id, tokenizer, ex_index))
    # with open("input_examples.txt", "w", encoding="utf-8") as f:
    #     f.writelines([str(feature) for feature in features])
    return features


def generate_data_for_pred(cls_token, cls_token_at_end, cls_token_segment_id, tokens,
                           label_ids, mask_padding_with_zero, max_seq_length, pad_on_left, pad_token,
                           pad_token_label_id, pad_token_segment_id, pred_start, pred_tokens, sep_token,
                           sep_token_extra, sequence_a_segment_id, sequence_b_segment_id, tokenizer, ex_index):
    special_tokens_count = tokenizer.num_special_tokens_to_add() + len(pred_tokens) + 1
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        label_ids = label_ids[: (max_seq_length - special_tokens_count)]
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens += [sep_token]
    label_ids += [pad_token_label_id]
    tokens += pred_tokens
    label_ids += [pad_token_label_id] * len(pred_tokens)
    tokens += [sep_token]
    label_ids += [pad_token_label_id]
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
    segment_ids = [sequence_a_segment_id] * (len(tokens) - len(pred_tokens) - 1)
    segment_ids += [sequence_b_segment_id] * (len(pred_tokens) + 1)
    if cls_token_at_end:
        tokens += [cls_token]
        label_ids += [pad_token_label_id]
        segment_ids += [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    position_ids = list(range(len(input_ids)))
    position_ids[-(len(pred_tokens) + 1):] = range(pred_start, pred_start + len(pred_tokens) + 1)
    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        label_ids = ([pad_token_label_id] * padding_length) + label_ids
        position_ids = ([pad_token] * padding_length) + position_ids
    else:
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length
        position_ids += [pad_token] * padding_length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(position_ids) == max_seq_length
    if ex_index < 2:
        logger.info("*** Example ***")
        logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
        logger.info("position_ids: %s", " ".join([str(x) for x in position_ids]))
    if "token_type_ids" not in tokenizer.model_input_names:
        segment_ids = None
    return InputFeatures(
        input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids,
        position_ids=position_ids
    )


def convert_examples_to_features_e2e(
        examples: List[InputExampleE2E],
        label_list: List[str],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        sequence_b_segment_id=1,
        mask_padding_with_zero=True,
        gold_predicates=False,
        pred_sampling_prob=1.0
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers
    # TODO model argument parts separately
    split = examples[0].guid[:examples[0].guid.index("-")] if len(examples) > 0 else ""
    label_map = defaultdict(lambda: label_list.index("O"))
    for i, label in enumerate(label_list):
        label_map[label] = i
    # label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    use_eos_token = tokenizer.eos_token is not None
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        pred_first_subwords = []
        tokens = []
        label_ids = [[] for _ in example.pred]
        index = 1
        for i, word in enumerate(example.words):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            first_subword_index = len(tokens)
            if cls_token is not None and not cls_token_at_end:
                first_subword_index += 1
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                for j, pred in enumerate(example.pred):
                    if pred == i + 1:
                        pred_first_subwords.append(first_subword_index)
                    label_ids[j].extend(
                        [label_map[example.labels[i][j]]] + [pad_token_label_id] * (len(word_tokens) - 1))
                index += 1

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = [labels[:max_seq_length - special_tokens_count] for labels in label_ids]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        if use_eos_token:
            tokens += [tokenizer.eos_token]
            for i in range(len(label_ids)):
                label_ids[i] += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            for i in range(len(label_ids)):
                label_ids[i] += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * (len(tokens))
        if cls_token is not None:
            if cls_token_at_end:
                tokens += [cls_token]
                for i in range(len(label_ids)):
                    label_ids[i] += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                for i in range(len(label_ids)):
                    label_ids[i] = [pad_token_label_id] + label_ids[i]
                segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        position_ids = list(range(len(input_ids)))
        # Zero-pad up to the sequence length.

        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            for i in range(len(label_ids)):
                label_ids[i] = ([pad_token_label_id] * padding_length) + label_ids[i]
            position_ids = ([pad_token] * padding_length) + position_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            for i in range(len(label_ids)):
                label_ids[i] += [pad_token_label_id] * padding_length
            position_ids += [pad_token] * padding_length
        other = label_map["O"]
        dummy_label_line = np.array(label_ids[0] if len(label_ids) > 0 else [pad_token_label_id] * max_seq_length)
        dummy_label_line[np.array(dummy_label_line) >= 0] = other
        tmp_labels = []
        i = 0
        for j in range(max_seq_length):
            if len(example.pred) > i and pred_first_subwords[i] == j:
                tmp_labels.append(label_ids[i])
                i += 1
            elif dummy_label_line[j] >= 0 and not gold_predicates and (split != "train" or random.random() < pred_sampling_prob):
                tmp_labels.append(dummy_label_line)
            else:
                tmp_labels.append([pad_token_label_id] * len(dummy_label_line))
        label_ids = tmp_labels
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        for i in range(len(label_ids)):
            assert len(label_ids[i]) == max_seq_length
        assert len(position_ids) == max_seq_length

        if ex_index <= 50:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids[0]]))
            logger.info("position_ids: %s", " ".join([str(x) for x in position_ids]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids,
                position_ids=position_ids
            )
        )
    return features


def get_labels(path: str):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def align_predictions(preds: np.ndarray, label_ids: np.ndarray, label_map) -> Tuple[List[int], List[int]]:
    batch_size, seq_len = preds.shape
    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] >= 0:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list


def get_system_predicates(preds: list, example, prediction_index, end2end):
    predicates = []
    if end2end:
        pred = preds[prediction_index]
        for i in range(len(example.words)):
            if np.any(np.array(pred)[i::len(example.words)] != "O"):
                predicates.append(i + 1)
    else:
        for i in range(len(example.words)):
            pred = preds[prediction_index + i]
            if np.any(np.array(pred) != "O"):
                predicates.append(i + 1)

    return predicates


def output_prop(preds: list, examples: List[InputExampleE2E], f, simulate_gold_preds, end2end):
    j = 0
    prediction_index = 0
    for ex_index, example in enumerate(examples):
        prediction = preds[prediction_index]
        max_len = int(math.sqrt(len(prediction)))
        sentence_len = min(max_len, len(example.words))
        expected_len = len(example.words) ** 2 if end2end else len(example.words)
        if len(prediction) < expected_len:
            prediction += ["O"] * (expected_len - len(prediction))
        predicates = example.pred if simulate_gold_preds else get_system_predicates(preds, example, prediction_index, end2end)
        prev_label = [None] * len(predicates)
        prev_output = [None] * len(predicates)
        for word_index, word in enumerate(example.words):
            i = word_index * sentence_len
            if word_index in example.pred:
                f.write(example.pred_lemma[example.pred.index(word_index)])
            elif word_index > 0:
                f.write("-")
            prev_pred = 1
            for pred_index, pred in enumerate(predicates):
                if not simulate_gold_preds:
                    i += pred - prev_pred # skip non-predicate labels
                if end2end:
                    actual_prediction = prediction[i]
                else:
                    index = pred - 1 if not simulate_gold_preds else pred_index
                    prediction = preds[prediction_index + index]
                    if len(prediction) < expected_len:
                        prediction += ["O"] * (expected_len - len(prediction))
                    actual_prediction = prediction[word_index]
                prev_pred = pred

                if actual_prediction == "O":
                    output = "*"
                    label = "O"
                else:
                    output = actual_prediction.replace("B-", "(").replace("I-", "").replace("_", "-") + "*"
                    label = actual_prediction.replace("B-", "").replace("I-", "")
                    if label != prev_label[pred_index] and output[0] != "(": # add begin tag if not present
                        output = "(" + output
                    if output[0] != "(":
                        output = "*"
                if prev_output[pred_index] is not None and (prev_label[pred_index] != label or output.startswith("(")) and prev_label[pred_index] != "O":
                    prev_output[pred_index] += ")"
                if prev_output[pred_index] is not None:
                    f.write("\t" + prev_output[pred_index])
                prev_label[pred_index] = label
                prev_output[pred_index] = output
            if simulate_gold_preds:
                i += 1
            if word_index > 0:
                f.write("\n")
                j += 1
        if len(example.words) in example.pred:
            f.write(example.pred_lemma[example.pred.index(len(example.words))])
        else:
            f.write("-")
        for pred_index, label in enumerate(prev_output):
            if label is not None:
                if prev_label[pred_index] != "O":
                    label += ")"
                f.write("\t" + label)
        f.write("\n\n")
        if end2end:
            prediction_index += 1
        else:
            prediction_index += len(example.pred) if simulate_gold_preds else len(example.words)
        j += 2

def eval_prop(gold_path, pred_path, data_dir):
    cmd = ["perl", os.path.join(data_dir, "srl-eval.pl"), gold_path, pred_path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    if stderr is not None:
        logger.error(stderr)
    logger.info("Official result for {}".format(pred_path))
    logger.info(stdout)
    return stdout


def eval_srl(y_true, y_pred, pred_probs, simulate_gold_preds=False, lang="cs", official=False, split=Split.test, end2end=True):
    if len(y_true.shape) >= 3:
        y_true = torch.transpose(y_true, 1, 2)
        y_pred = torch.transpose(y_pred, 1, 2)
    y_true = torch.reshape(y_true, [y_true.shape[0], -1])
    y_pred = torch.reshape(y_pred, [y_pred.shape[0], -1])
    data_dir = os.path.join(SRL_DATA_DIR, lang)
    labels = get_labels(os.path.join(data_dir, "labels.txt"))
    labels_map = {i: label for i, label in enumerate(labels)}
    y_pred, y_true = align_predictions(y_pred.detach().numpy(), y_true.detach().numpy(), labels_map)
    if official:
        examples = read_examples_from_ontonotes(data_dir, split)
        with tempfile.NamedTemporaryFile(delete=True, mode="w") as prediction_file:
            output_prop(y_pred, examples, prediction_file, simulate_gold_preds, end2end)
            prediction_file.flush()
            report = eval_prop(os.path.join(data_dir, split.value + ".prop"), prediction_file.name, data_dir)
            if report != "":
                line = report.split("\n")[6].split()
                prec, rec, f1 = float(line[4]), float(line[5]), float(line[6])
            else:
                prec = rec = f1 = 0.0
            return {"accuracy": f1, "precision": prec, "recall": rec, "f1": f1, "report": report}
    else:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "report": classification_report(y_true, y_pred)
        }


def print_eval_srl(result, use_wandb, desc):
    print(70 * '*-')
    print(desc + "SRL report")
    print(desc + " F1:{:.4f}".format(result["f1"]))
    print(desc + " accuracy:{:.4f}".format(result["accuracy"]))
    print(desc + " precision:{:.4f}".format(result["precision"]))
    print(desc + " recall:{:.4f}".format(result["recall"]))
    print(desc + " classification report")
    print(result["report"])
    print(70 * '*-')

    if use_wandb is True:
        try:
            wandb.log({'f1_' + desc: result["f1"], 'accuracy_' + desc: result["accuracy"],
                       'precision_' + desc: result["precision"], 'recall_' + desc: result["recall"]}, commit=False)
            wandb.run.summary['f1_' + desc] = result["f1"]
            wandb.run.summary['accuracy_' + desc] = result["accuracy"]
            wandb.run.summary['precision_' + desc] = result["precision"]
            wandb.run.summary['recall_' + desc] = result["recall"]
            wandb.run.summary['report_' + desc] = result['report']
        except Exception as e:
            logger.error("Error WANDB with exception e:" + str(e))

    results_head = '\tF1_' + str(desc) + '\t' + 'Accuracy_' + str(desc) + '\t' + 'Precision_' + str(
        desc) + '\t' + 'Recall_' + str(desc)
    results = f'{result["f1"]}\t{result["accuracy"]}\t{result["precision"]}\t{result["recall"]}\t\t\t'

    return results_head, results


def eval_pred(file):
    closed = []
    label_index = 0
    with open(file, encoding="utf-8") as f:
        for line_index, line in enumerate(f.read().splitlines()):
            fields = line.split()
            labels = fields[1:]
            if len(labels) != len(closed):
                if len(closed) > 0 and not all(closed):
                    print(f"unclosed at the end of the sentence on the line: {line_index}, label index: {label_index}")
                    print(line)
                closed = [not (label.startswith("(") and not label.endswith(")")) for label in labels]
            else:
                for index, label in enumerate(labels):
                    if label.startswith("("):
                        if not closed[index]:
                            print(f"Opened but not closed on the line: {line_index}, label index: {label_index}")
                            print(line)
                        closed[index] = False
                    if label.endswith(")"):
                        if closed[index]:
                            print(f"Closed but not opened on the line: {line_index}, label index: {label_index}")
                            print(line)
                        closed[index] = True
            label_index += len(fields)

def gold_self_eval(file):
    buffer = []
    sentence_index = 0
    data_dir = '/data/ondfa@gapps.zcu.cz/IdeaProjects/aspect-based-sentiment/data/datasets/srl/en'
    with open(file, encoding="utf-8") as f:
        for line_index, line in enumerate(f.read().splitlines()):
            if line == "":
                file = tempfile.NamedTemporaryFile(delete=True, mode="w")
                file.writelines(buffer)
                file.flush()
                output = eval_prop(file.name, file.name, data_dir)
                if output == "":
                    print("Failed to eval: " + str(sentence_index))
                elif int(output.split("\n")[6].split()[2]) > 0:
                    print(f"sentence_index: {sentence_index}, line index: {line_index}")
                sentence_index += 1
                buffer = []
            else:
                buffer.append(line + "\n")


if __name__ == '__main__':
    # eval_pred('/tmp/tmpen9cod5i')
    gold_self_eval("/data/ondfa@gapps.zcu.cz/IdeaProjects/aspect-based-sentiment/data/datasets/srl/en/ontonotes-srl-test_from_skel.prop")


