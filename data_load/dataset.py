import os.path

import torch
from torch.utils.data import Dataset, DataLoader
# import utils_srl
from config import INJECTION_MODE_MULTI_TASK, SRL_DATA_DIR
from utils import utils_srl

SRL_SOURCE = "srl_source"
ABSA_SOURCE = "absa_source"

class SRL_Dataset(Dataset):
    def __init__(self, srl_data, tokenizer, max_len, lang, simulate_gold_preds=False, predicate_sampling_prob=1.0):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.simulate_gold_preds = simulate_gold_preds
        self.predicate_sampling_prob = predicate_sampling_prob
        self.data_dir = os.path.join(SRL_DATA_DIR, lang)
        self.prepare_srl(srl_data)
        self.length = len(self.srl_features)
        self.srl_len = len(self.srl_features)


    def prepare_srl(self, srl_data):
        labels = utils_srl.get_labels(os.path.join(self.data_dir, "labels.txt"))
        self.srl_features = utils_srl.convert_examples_to_features(
            srl_data,
            labels,
            self.max_len,
            self.tokenizer,
            cls_token_at_end=False,
            # xlnet has a cls token at the end
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=False,
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(self.tokenizer.padding_side == "left"),
            pad_token=self.tokenizer.pad_token_id,
            pad_token_segment_id=self.tokenizer.pad_token_type_id,
            pad_token_label_id=-100,
            gold_predicates=self.simulate_gold_preds,
            pred_sampling_prob=self.predicate_sampling_prob
        )

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.srl_features[item].token_type_ids is None:
            return {"input_ids": torch.tensor(self.srl_features[item].input_ids),
                    "attention_mask": torch.tensor(self.srl_features[item].attention_mask),
                    "position_ids": torch.tensor(self.srl_features[item].position_ids),
                    "labels": torch.tensor(self.srl_features[item].label_ids),
                    "source": SRL_SOURCE}
        else:
            return {"input_ids": torch.tensor(self.srl_features[item].input_ids),
                    "attention_mask": torch.tensor(self.srl_features[item].attention_mask),
                    "position_ids": torch.tensor(self.srl_features[item].position_ids),
                    "token_type_ids": torch.tensor(self.srl_features[item].token_type_ids),
                    "labels": torch.tensor(self.srl_features[item].label_ids),
                    "source": SRL_SOURCE}

class SRL_Dataset_E2E(Dataset):
    def __init__(self, srl_data, tokenizer, max_len, lang, simulate_gold_preds=False, predicate_sampling_prob=0.0):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.simulate_gold_preds = simulate_gold_preds
        self.lang = lang
        self.data_dir = os.path.join(SRL_DATA_DIR, lang)
        self.predicate_sampling_prob = predicate_sampling_prob
        self.prepare_srl(srl_data)
        self.length = len(self.srl_features)
        self.srl_len = len(self.srl_features)


    def prepare_srl(self, srl_data):
        labels = utils_srl.get_labels(os.path.join(self.data_dir, "labels.txt"))
        self.srl_features = utils_srl.convert_examples_to_features_e2e(
            srl_data,
            labels,
            self.max_len,
            self.tokenizer,
            cls_token_at_end=False,
            # xlnet has a cls token at the end
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=False,
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(self.tokenizer.padding_side == "left"),
            pad_token=self.tokenizer.pad_token_id,
            pad_token_segment_id=self.tokenizer.pad_token_type_id,
            pad_token_label_id=-100,
            gold_predicates=self.simulate_gold_preds,
            pred_sampling_prob=self.predicate_sampling_prob
        )

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.srl_features[item].token_type_ids is None:
            return {"input_ids": torch.tensor(self.srl_features[item].input_ids),
                    "attention_mask": torch.tensor(self.srl_features[item].attention_mask),
                    "position_ids": torch.tensor(self.srl_features[item].position_ids),
                    "labels": torch.tensor(self.srl_features[item].label_ids),
                    "source": SRL_SOURCE}
        else:
            return {"input_ids": torch.tensor(self.srl_features[item].input_ids),
                    "attention_mask": torch.tensor(self.srl_features[item].attention_mask),
                    "position_ids": torch.tensor(self.srl_features[item].position_ids),
                    "token_type_ids": torch.tensor(self.srl_features[item].token_type_ids),
                    "labels": torch.tensor(self.srl_features[item].label_ids),
                    "source": SRL_SOURCE}


def build_ids_tensors(tokenizer, text_help, text_orig, max_len):
    ids_tensor_help = tokenizer(text_help,
                                     max_length=max_len,
                                     add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                     return_token_type_ids=False,
                                     padding='max_length',
                                     return_attention_mask=True,
                                     truncation=True,
                                     return_tensors='pt',  # Return PyTorch tensors
                                     )

    ids_tensor_orig = tokenizer(text_orig,
                                     max_length=max_len,
                                     add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                     return_token_type_ids=False,
                                     padding='max_length',
                                     return_attention_mask=True,
                                     truncation=True,
                                     return_tensors='pt',  # Return PyTorch tensors
                                     )

    ids_tensor = tokenizer(text_help, text_orig,
                                max_length=max_len,
                                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                return_token_type_ids=True,
                                padding='max_length',
                                return_attention_mask=True,
                                truncation=True,
                                return_tensors='pt',  # Return PyTorch tensors
                                )

    return ids_tensor_help, ids_tensor_orig, ids_tensor


class ABSA_QA_NLI_Dataset(Dataset):
    def __init__(self, ids, texts_help, texts_orig, labels, tokenizer, max_len):
        self.ids = ids
        self.texts_help = texts_help
        self.texts_orig = texts_orig
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        if len(texts_help) != len(ids):
            raise Exception(f"Lens of texts_a: {len(texts_help)} does not correspond to lengths of ids: {len(ids)}")

        if len(texts_help) != len(texts_orig):
            raise Exception(f"Lens of texts_a: {len(texts_help)} does not correspond to lengths of texts_b: {len(texts_orig)}")

        if len(texts_help) != len(labels):
            raise Exception(f"Lens of texts_a: {len(texts_help)} does not correspond to lengths of labels: {len(labels)}")

    def __len__(self):
        return len(self.texts_help)

    def __getitem__(self, item):
        text_help = str(self.texts_help[item])
        text_orig = str(self.texts_orig[item])
        label = self.labels[item]

        ids_tensor_help, ids_tensor_orig, ids_tensor = build_ids_tensors(self.tokenizer, text_help, text_orig,
                                                                         self.max_len)

        return {
            'text_help': text_help,
            'text_orig': text_orig,
            'input_ids_help': ids_tensor_help['input_ids'].flatten(),
            'input_ids_orig': ids_tensor_orig['input_ids'].flatten(),
            'input_ids': ids_tensor['input_ids'].flatten(),
            'token_type_ids': ids_tensor['token_type_ids'].flatten(),
            'attention_mask_help': ids_tensor_help['attention_mask'].flatten(),
            'attention_mask_orig': ids_tensor_orig['attention_mask'].flatten(),
            'attention_mask': ids_tensor['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            "source": ABSA_SOURCE
        }


def build_data_loader(data_df, tokenizer, args, shuffle=False, srl_data=None):
    data_loader = None
    if args.task == 'CAT':
        if args.injection_mode == INJECTION_MODE_MULTI_TASK:
            data_loader = build_data_loader_srl_absa(data_df[0], data_df[1], tokenizer, args, shuffle)
        else:
            data_loader = build_data_loader_cat(data_df, tokenizer, args, shuffle)

    elif args.task == 'TERM':
        raise NotImplementedError("Not implemented yet, do it now!")
        pass
    elif args.task == 'SRL':
        if args.end2end:
            data_loader = build_data_loader_srl_e2e(data_df, tokenizer, args, shuffle)
        else:
            data_loader = build_data_loader_srl(data_df, tokenizer, args, shuffle)

    elif args.task == 'PRE-TRAIN':
        # pro inifnite data nebo stream data lze pouzit IterableDataset
        # https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
        raise NotImplementedError("Not implemented yet, do it now!")
        pass
    else:
        raise Exception('Unknown task:' + str(args.task))

    return data_loader


def build_data_loader_srl_absa(absa_data_df, srl_data, tokenizer, args, shuffle=False):
    if srl_data == None:
        raise Exception("The srl data are None")
    absa_loader = build_data_loader_cat(absa_data_df, tokenizer, args, shuffle)
    srl_loader = build_data_loader_srl_e2e(srl_data, tokenizer, args, shuffle)
    data_loader = ABSA_SRL_DataLoader(absa_loader, srl_loader, args.batch_size)

    return data_loader

    # set num_workers to 0 while in debug
    # return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.data_loader_num_workers, shuffle=shuffle)


class ABSA_SRL_DataLoader(object):
    def __init__(self, absa_data_loader, srl_data_loader, batch_size):
        self.absa_data_loader = absa_data_loader
        self.srl_data_loader = srl_data_loader
        self.absa_is_smaller = len(absa_data_loader) < len(srl_data_loader)
        self.use_srl = False
        self.batch_size = batch_size

    def __len__(self):
        return 2 * len(self.srl_data_loader) if self.absa_is_smaller else 2 * len(self.absa_data_loader)

    def __next__(self):
        if self.use_srl:
            example = next(self.srl_iter, None)
            if example is None:
                if not self.absa_is_smaller:
                    self.srl_iter = iter(self.srl_data_loader)
                    example = next(self.srl_iter, None)
                else:
                    example = next(self.srl_iter)
        else:
            example = next(self.absa_iter, None)
            if example is None:
                if self.absa_is_smaller:
                    self.absa_iter = iter(self.absa_data_loader)
                    example = next(self.absa_iter, None)
                else:
                    example = next(self.absa_iter)

        self.use_srl = not self.use_srl
        return example

    def __iter__(self):
        self.absa_iter = iter(self.absa_data_loader)
        self.srl_iter = iter(self.srl_data_loader)
        self.use_srl = False
        return self


def build_data_loader_srl(data, tokenizer, args, shuffle=False):
    dataset = SRL_Dataset(data, tokenizer, args.max_seq_len, args.dataset_lang, args.simulate_gold_preds, args.predicate_sampling_prob)
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.data_loader_num_workers, shuffle=shuffle)

def build_data_loader_srl_e2e(data, tokenizer, args, shuffle=False):
    dataset = SRL_Dataset_E2E(data, tokenizer, args.max_seq_len, args.dataset_lang, args.simulate_gold_preds, args.predicate_sampling_prob)
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.data_loader_num_workers, shuffle=shuffle)

def build_data_loader_cat(data_df, tokenizer, args, shuffle=False):
    sol_type = args.solution_type_cat
    dataset = None
    if sol_type == 'QA_M' or sol_type == 'QA_B' or sol_type == 'NLI_M' or sol_type == 'NLI_B':
        dataset = ABSA_QA_NLI_Dataset(data_df.id.to_numpy(),
                                      data_df.help_sentence.to_numpy(),
                                      data_df.orig_sentence.to_numpy(),
                                      data_df.label.to_numpy(),
                                      tokenizer,
                                      args.max_seq_len)
    else:
        raise Exception('Unsupported type of solution:' + str(sol_type))

    # set num_workers to 0 while in debug
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.data_loader_num_workers, shuffle=shuffle)

