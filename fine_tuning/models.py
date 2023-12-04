import torch
from torch import nn
import logging
import torch.nn.functional as F

from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT, INJECTION_MODE_CONCAT_AVERAGE, INJECTION_MODE_CONCAT_CONVOLUTION

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)

class SrlBertOld(nn.Module):

    def __init__(self, base_model, num_labels, args):
        super().__init__()
        if num_labels is None:
            num_labels = base_model.config.num_labels
        self.num_labels = num_labels
        self.projection = torch.nn.Linear(3 * base_model.config.hidden_size, base_model.config.hidden_size)
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(args.classifier_dropout)
        self.out = torch.nn.Linear(base_model.config.hidden_size, num_labels, bias=not args.disable_output_bias)
        self.base_model = base_model
        self.config = base_model.config
        weights = torch.ones(num_labels)
        weights[62] = args.srl_other_weight
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def forward(self, *args, **kwargs):
        if "labels" in kwargs:
            labels = kwargs['labels']
            del kwargs["labels"]
        else:
            labels = None
        hidden = self.base_model(*args, **kwargs)
        seq_out = hidden.last_hidden_state
        # seq_out = hidden[0]
        seq_out = torch.transpose(seq_out, 1, 2)
        pred_arg = torch.matmul(torch.unsqueeze(seq_out, -1), torch.unsqueeze(seq_out, -2))
        seq_out = torch.transpose(seq_out, 1, 2)
        # add unary representations
        pred_arg = torch.transpose(pred_arg, 1, 3)
        pred_arg = torch.transpose(pred_arg, 1, 2)
        s = pred_arg.shape
        pred_arg = torch.cat([pred_arg, torch.unsqueeze(seq_out, -2).expand(s)], -1)
        pred_arg = torch.cat([pred_arg, torch.unsqueeze(seq_out, -3).expand(s)], -1)
        proj = self.projection(pred_arg)
        proj = self.activation(proj)
        proj = self.dropout(proj)
        out = self.out(proj)
        out = out.transpose(1, -1)
        if labels is not None:
            loss = self.loss_fn(out, labels)
            return loss, out
        else:
            return out


class SrlBert(nn.Module):

    def __init__(self, base_model, num_labels, args):
        super().__init__()
        if num_labels is None:
            num_labels = base_model.config.num_labels
        self.num_labels = num_labels
        self.use_dotprod = args.srl_use_dot_product
        self.separate_encoding = args.srl_separate_encoding
        self.seq_len = args.max_seq_len
        repre_size = 2 * base_model.config.hidden_size
        if self.use_dotprod:
            repre_size += base_model.config.hidden_size
        self.projection = torch.nn.Linear(repre_size, base_model.config.hidden_size)
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(args.classifier_dropout)
        self.out = torch.nn.Linear(base_model.config.hidden_size, num_labels, bias=not args.disable_output_bias)
        self.base_model = base_model
        self.config = base_model.config
        weights = torch.ones(num_labels)
        weights[62] = args.srl_other_weight
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def forward(self, *args, **kwargs):
        if "labels" in kwargs:
            labels = kwargs['labels']
            del kwargs["labels"]
        else:
            labels = None
        if self.separate_encoding:
            kwargs = self.repeat_inputs(kwargs)
        hidden = self.base_model(*args, **kwargs)
        seq_out = hidden.last_hidden_state
        if self.separate_encoding:
            pred_arg = self.bitext_model(seq_out[:, :self.seq_len, :], seq_out[:, self.seq_len:, :])
        else:
            pred_arg = self.bitext_model(seq_out, seq_out)
        proj = self.projection(pred_arg)
        proj = self.activation(proj)
        proj = self.dropout(proj)
        out = self.out(proj)
        out = out.transpose(1, -1)
        if labels is not None:
            loss = self.loss_fn(out, labels)
            return loss, out
        else:
            return out

    def repeat_inputs(self, kwargs):
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        position_ids = torch.unsqueeze(torch.arange(0, input_ids.shape[1]), 0).expand(input_ids.shape).to(input_ids.device)
        token_type_ids = torch.cat([torch.zeros_like(input_ids), torch.ones_like(input_ids)], -1).to(input_ids.device)
        kwargs["input_ids"] = self.repeat_input(input_ids)
        kwargs["attention_mask"] = self.repeat_input(attention_mask)
        kwargs["position_ids"] = self.repeat_input(position_ids)
        kwargs["token_type_ids"] = token_type_ids
        return kwargs

    def repeat_input(self, input):
        return torch.cat([input, input], -1)

    def bitext_model(self, first, second):
        s = [first.shape[0], first.shape[1], first.shape[1], first.shape[2]]
        pred_arg = torch.cat([torch.unsqueeze(first, -3).expand(s), torch.unsqueeze(second, -2).expand(s)], -1)
        if self.use_dotprod:
            first = torch.transpose(first, 1, 2)
            second = torch.transpose(second, 1, 2)
            dot = torch.matmul(torch.unsqueeze(first, -1), torch.unsqueeze(second, -2))
            dot = torch.transpose(dot, 1, 3)
            dot = torch.transpose(dot, 1, 2)
            pred_arg = torch.cat([dot, pred_arg], -1)
        return pred_arg


class SrlBertSimple(nn.Module):

    def __init__(self, base_model, args):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        if "labels" in kwargs:
            labels = kwargs['labels']
            del kwargs["labels"]
        else:
            labels = None
        hidden = self.base_model(*args, **kwargs)
        out = hidden.logits
        out = out.transpose(1, -1)
        if labels is not None:
            loss = self.loss_fn(out, labels)
            return loss, out
        else:
            return out

class SRLAverageModel(nn.Module):
    """
    Průměruje parametry dvou modelů, sečte parametry z obou modelů a součet nastaví
    ABSA modelu

    args.use_custom_model nesmi byt nastaven
    """

    def __init__(self, absa_model, srl_model):
        super(SRLAverageModel, self).__init__()
        self.absa_model = absa_model
        self.srl_model = srl_model
        self.absa_model_layers = list(absa_model.named_parameters())
        self.srl_model_layer = list(srl_model.named_parameters())

    def forward(self, *args, **kwargs):
        for i, param in enumerate(self.srl_model.base_model.named_parameters()):
            absa_param = (self.absa_model_layers[i])[1]
            srl_param = param[1]

            with torch.no_grad():
                absa_param.add_(srl_param)

        return self.absa_model(*args, **kwargs)

    def set_dropout(self, classifier_dropout, hidden_dropout_prob):
        self.srl_model.config.hidden_dropout_prob = hidden_dropout_prob
        self.absa_model.config.hidden_dropout_prob = hidden_dropout_prob

        self.absa_model.config.hidden_dropout_prob = classifier_dropout

class SRLConcatModel(nn.Module):
    """

    """

    def __init__(self, absa_model, srl_model, args, num_labels):
        super(SRLConcatModel, self).__init__()
        filter_sizes = [3, 4, 5]
        self.num_labels = num_labels
        self.absa_model = absa_model.base_model
        self.srl_model = srl_model


        freeze_base_model(srl_model)
        self.mode = args.injection_mode
        self.config = absa_model.config
        self.enable_masking = args.enable_masking
        model_hidden_size = self.config.hidden_size
        model_max_seq = self.config.max_position_embeddings
        num_filters = int(model_hidden_size / 2)

        if self.mode == INJECTION_MODE_CONCAT_CONVOLUTION:
            self.conv1d_list = nn.ModuleList([
                nn.Conv1d(in_channels=model_hidden_size, out_channels=num_filters, kernel_size=filter_size)
                for filter_size in filter_sizes
            ])
            self.new_pool_input_size = num_filters * len(filter_sizes) + model_hidden_size
            self.new_pool_output_size = model_hidden_size
        elif self.mode == INJECTION_MODE_CONCAT_AVERAGE:
            self.new_pool_input_size = 2 * model_hidden_size
            self.new_pool_output_size = model_hidden_size
        else:
            raise Exception("Unknown mode:" + str(self.mode))

        self.dense = nn.Linear(self.new_pool_input_size, self.new_pool_output_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(args.classifier_dropout)
        self.classifier = nn.Linear(self.new_pool_output_size, self.num_labels)

    def forward(self, *args, **kwargs):

        labels = kwargs['labels']
        del kwargs['labels']
        out_absa = self.absa_model(*args, **kwargs)
        cls_absa_orig = out_absa.last_hidden_state[:,0]
        # cls_absa_orig = out_absa[0][:,0,:]

        out_srl = self.srl_model(*args, **kwargs)
        srl_output = out_srl.last_hidden_state

        if self.enable_masking is True:
            attention_mask = kwargs['attention_mask']
            attention_mask = torch.unsqueeze(attention_mask, 2)
            srl_output = srl_output * attention_mask

        # srl_output = out_srl[0]
        if self.mode == 'none':
            cls_absa_orig = torch.unsqueeze(cls_absa_orig, 1)
            merged_out = torch.cat((cls_absa_orig, srl_output), dim=1)
        elif self.mode == 'concat-avg':
            srl_output = torch.mean(srl_output, 1)
            merged_out = torch.cat((cls_absa_orig, srl_output), dim=-1)
        elif self.mode == INJECTION_MODE_CONCAT_CONVOLUTION:
            x_reshaped = srl_output.permute(0, 2, 1)
            x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
            x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]
            x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
            merged_out = torch.cat((cls_absa_orig, x_fc), dim=-1)
        else:
            raise Exception("Unknown mode:" + str(self.mode))

        loss_fn = torch.nn.CrossEntropyLoss()
        hidden = self.dense(merged_out)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        output = self.classifier(hidden)
        loss = loss_fn(output, labels)

        return loss, output


class MultiTaskModel(nn.Module):

    def __init__(self, base_model, absa_num_labels, srl_num_labels, args):
        super(MultiTaskModel, self).__init__()
        self.srl_num_labels = srl_num_labels
        self.absa_num_labels = absa_num_labels
        self.base_model = base_model.base_model
        self.config = base_model.config
        model_hidden_size = self.config.hidden_size

        # absa head
        self.dense = nn.Linear(model_hidden_size, model_hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(args.classifier_dropout)
        self.classifier_absa = nn.Linear(model_hidden_size, self.absa_num_labels)

        # srl head
        # self.classifier_srl = nn.Linear(model_hidden_size, self.srl_num_labels)
        self.srl_model = SrlBert(self.base_model, self.srl_num_labels, args)

    def forward(self, *args, **kwargs):
        labels = kwargs['labels']
        # Predpokladame ze v jednom batchi jsou vzdy data jen pro jeden task
        if len(labels.shape) == 1:
            del kwargs['labels']
            loss_fn = torch.nn.CrossEntropyLoss()
            preds = self.base_model(*args, **kwargs)
            hidden = self.dense(preds.last_hidden_state[:, 0])
            # hidden = self.dense(preds[0][:, 0, :])
            hidden = self.activation(hidden)
            hidden = self.dropout(hidden)
            output = self.classifier_absa(hidden)
            loss = loss_fn(output, labels)
        else:
            loss, output = self.srl_model(*args, **kwargs)
            # output = self.dropout(preds.last_hidden_state)
            # output = self.dropout(preds[0])
            # output = self.classifier_srl(output)

            # nase implementace je treba z (8,512,125) udelat (8,125,512) protoze Cross entropy v torchi
            # ocekava ze tam bude (Batch_size, pocet_trid, dalsi_dimenze)
            # output = output.transpose(1,-1)


            # Tohle je origo implementace jak je to udelany v BERTovi
            # see https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/bert/modeling_bert.py#L1698
            # Je to tam udelane takhle v transformers
            # logits = output
            # attention_mask = kwargs['attention_mask']
            # if attention_mask is not None:
            #     active_loss = attention_mask.view(-1) == 1
            #     active_logits = logits.view(-1, self.srl_num_labels)
            #     active_labels = torch.where(
            #         active_loss, labels.view(-1), torch.tensor(loss_fn.ignore_index).type_as(labels)
            #     )
            #     loss = loss_fn(active_logits, active_labels)
            # else:
            #     loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))


        return loss, output






def unfreeze_base_model(model):
    # logger.info("Freezing base model layers, for model:" + str(model))
    for name, param in model.base_model.named_parameters():
        param.requires_grad = True


def freeze_base_model(model):
    # logger.info("Freezing base model layers, for model:" + str(model))
    for name, param in model.base_model.named_parameters():
        param.requires_grad = False