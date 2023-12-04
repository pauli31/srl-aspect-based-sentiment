import logging
import time
from collections import defaultdict

import numpy as np
import torch
import wandb
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

import utils.utils_srl
from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT
from data_load.dataset import SRL_SOURCE, ABSA_SOURCE
from fine_tuning.evaluation import print_category_results
from utils.general_utils import format_time

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)

FAST_DEBUG = False
NUM_FAST_DEBUG_ITER = 100


def run_training(args, model, device, loss_fn, optimizer, scheduler, tuner,
                 train_data_loader, val_data_loader, dev_target_data_loader=None):
    validation_size = 0
    if tuner is not None:
        validation_size = tuner.dev_size
    num_epochs = args.epoch_num

    logger.info('\ndevice: {0:s}\n'.format(device.type))

    history = defaultdict(list)
    best_accuracy = 0

    t00 = time.time()
    for epoch in range(num_epochs):
        logger.info('')
        logger.info('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epochs))
        t0 = time.time()
        logger.info('-' * 30)

        train_acc, train_loss = train_epoch(args, model, loss_fn, optimizer, scheduler, device, epoch,
                                            train_data_loader)

        train_time = format_time(time.time() - t0)
        logger.info(f'Total train time for epoch:{train_time}')
        logger.info(f'Train loss: {train_loss} accuracy: {train_acc}')

        t0 = time.time()

        if val_data_loader is not None and validation_size > 0:
            texts_orig, texts_help, y_pred, y_pred_probs, y_true,\
            val_loss, val_acc, y_pred_srl, y_pred_probs_srl,\
                y_true_srl = get_predictions(model, val_data_loader, device, args.batch_size, args, loss_fn)
            if len(y_pred) > 0:
                dev_results = tuner.eval_model(y_true, y_pred, y_pred_probs)
                print_category_results("dev", dev_results, False, args)
            elif len(y_pred_srl) > 0:
                dev_results = utils.utils_srl.eval_srl(y_true_srl, y_pred_srl, y_pred_probs_srl, lang=args.dataset_lang)
                utils.utils_srl.print_eval_srl(dev_results, args.enable_wandb, "SRL-dev")

        else:
            val_acc = val_loss = 0

        val_time = format_time(time.time() - t0)
        logger.info(f'Val   loss {val_loss} accuracy {val_acc}')
        logger.info(f'Total validation time for epoch: {val_time}')

        if dev_target_data_loader is not None:
            # target_val_acc, target_val_loss = eval_model()
            target_val_acc = target_val_loss = 0
            logger.info(f'Target language val   loss {target_val_loss} accuracy {target_val_acc}')
        else:
            target_val_acc = target_val_loss = 0

        if args.enable_wandb is True:
            try:
                wandb.log({'train_acc': train_acc, 'train_loss': train_loss, 'val_acc': val_acc, 'val_loss': val_loss,
                           'train_time': train_time, 'val_time': val_time, 'target_val_acc': target_val_acc ,
                           'target_val_loss': target_val_loss})
            except Exception as e:
                logger.error("Error WANDB with exception e: " + str(e))

        # if val_acc > best_accuracy:
        #     torch.save(model.state_dict(), 'best_model_state.bin')
        #     best_accuracy = val_acc

        total_time = time.time() - t00
        logger.info(f'Total time for one epoch including validation time: {total_time}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['val_target_acc'].append(val_acc)
        history['val_target_loss'].append(val_loss)
        history['train_time'].append(train_time)
        history['val_time'].append(val_time)
        history['total_time'].append(total_time)

    return history


def train_epoch(args, model, loss_fn, optimizer, scheduler, device, epoch, data_loader):
    model = model.train()

    losses = []
    correct_pred_tmp = 0
    correct_pred_absa_tmp = 0
    correct_pred_srl_tmp = 0

    total_processed_predictions_absa_tmp = 0
    total_processed_predictions_srl_tmp = 0

    correct_predictions = 0
    total_processed_examples = 0
    total_processed_predictions = 0

    total_processed_predictions_absa = 0
    correct_predictions_absa = 0
    total_processed_predictions_srl = 0
    correct_predictions_srl = 0

    running_loss = 0.0
    n_processed_preds_tmp = 0

    # time since epoch started
    t0 = time.time()

    data_loader_len = len(data_loader)

    # the true number wil be little bit lower, bcs we do not align the data
    total_examples = data_loader_len * data_loader.batch_size

    batch_times = []

    # for i, batch in enumerate(tqdm(data_loader, desc="Iteration")):
    for i, batch in enumerate(data_loader):
        t_batch = time.time()
        if FAST_DEBUG is True:
            # only for testing purposes
            if i == NUM_FAST_DEBUG_ITER:
                break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        if "token_type_ids" in batch:
            token_type_ids = batch['token_type_ids'].to(device)
        else:
            token_type_ids = torch.zeros_like(input_ids).to(device)
        source = batch['source'][0]

        n_predicted_classes = len(labels.shape)
        # end2end 250 x 250
        if len(labels.shape) == 3:
            n_predicted_classes = labels.shape[1] * labels.shape[2]
        elif len(labels.shape) == 2:
            n_predicted_classes = labels.shape[1]

        # get size of batch
        data_size = len(labels)
        total_processed_examples += data_size
        input = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask, "labels": labels, "return_dict": True}
        if "position_ids" in batch:
            input["position_ids"] = batch["position_ids"].to(device)

        if args.use_custom_model:
            loss, logits = model(**input)
        else:
            input["labels"] = labels if loss_fn is None else None
            output_obj = model(**input)

            logits = output_obj.logits
            if loss_fn is None:
                loss = output_obj.loss
            else:
                loss = loss_fn(logits, labels)

        _, preds = torch.max(logits, dim=1)


        if source == SRL_SOURCE:
            tmp_labels = labels[labels >= 0]
            tmp_preds = preds[labels >= 0]
            tmp = torch.sum(tmp_preds == tmp_labels)
            tmp = tmp.detach().cpu().int().item()
            total_processed_predictions_srl += len(tmp_labels)
            correct_predictions_srl += tmp
            correct_pred_srl_tmp += tmp
            total_processed_predictions_srl_tmp += len(tmp_labels)
            total_processed_predictions += len(tmp_labels)
            n_processed_preds_tmp += len(tmp_labels)
        elif source == ABSA_SOURCE:
            tmp = torch.sum(preds == labels)
            tmp = tmp.detach().cpu().int().item()
            total_processed_predictions_absa += data_size
            correct_predictions_absa += tmp
            correct_pred_absa_tmp += tmp
            total_processed_predictions_absa_tmp += data_size * n_predicted_classes
            total_processed_predictions += data_size * n_predicted_classes
            n_processed_preds_tmp += data_size * n_predicted_classes
        else:
            raise Exception("Unknown source:" + str(source))
        correct_pred_tmp += tmp
        correct_predictions += tmp

        if args.data_parallel is True:
            # https://discuss.pytorch.org/t/loss-function-in-multi-gpus-training-pytorch/76765
            # https://discuss.pytorch.org/t/how-to-fix-gathering-dim-0-warning-in-multi-gpu-dataparallel-setting/41733
            # print("loss:" + str(loss))
            # loss_mean = loss.mean()
            # print("Loss mean:" + str(loss_mean))
            loss = loss.mean()

        loss_item = loss.item()
        losses.append(loss_item)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # store the batch time
        batch_times.append(time.time() - t_batch)

        # print statistics
        running_loss += loss_item
        if i % args.print_stat_frequency == 0 and not i == 0:
            # logger.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(i, total_processed_examples, elapsed))
            try:
                last_lr = scheduler.get_last_lr()
                last_lr = last_lr[0]
            except Exception as e:
                last_lr = 0
                logger.error("Cannot parse acutal learning rate")
            avg_batch_time = np.mean(batch_times)
            eta = (data_loader_len - i) * avg_batch_time

            avg_batch_time = format_time(avg_batch_time)
            eta = format_time(eta)

            elapsed = format_time(time.time() - t0)
            avg_loss = running_loss / args.print_stat_frequency
            acc = correct_pred_tmp / n_processed_preds_tmp

            absa_avg_acc_tmp = 0
            if total_processed_predictions_absa != 0:
                absa_avg_acc_tmp = correct_predictions_absa / total_processed_predictions_absa
            acc_absa = 0
            if total_processed_predictions_absa_tmp != 0:
                acc_absa = correct_pred_absa_tmp / total_processed_predictions_absa_tmp
            acc_srl = 0
            if total_processed_predictions_srl_tmp != 0:
                acc_srl = correct_pred_srl_tmp / total_processed_predictions_srl_tmp

            logger.info(f'Batch: {i:5}/{data_loader_len:<5} avg loss: {avg_loss:.4f} acc: {acc:.3f}'
                        f' processed: {total_processed_examples}/{total_examples} examples | epoch-time: {elapsed},'
                        f' eta:{eta} avg-batch-time:{avg_batch_time} actual lr: {last_lr:.8f} | acc absa:{acc_absa:.4f} acc srl:{acc_srl:.4f} | avg acc:{absa_avg_acc_tmp:.4f}')

            if args.enable_wandb is True:
                try:
                    wandb.log(
                        {'epoch': epoch, 'batch': i, 'avg_loss': avg_loss, 'avg_accuracy': acc, 'current_lr': last_lr,
                         'avg_absa_accuracy' : acc_absa, 'avg_srl_accuracy' : acc_srl})
                except Exception as e:
                    logger.error("Error WANDB with exception e:" + str(e))

            running_loss = 0.0
            n_processed_preds_tmp = 0.0
            correct_pred_tmp = 0.0
            correct_pred_absa_tmp = 0
            correct_pred_srl_tmp = 0
            total_processed_predictions_absa_tmp = 0
            total_processed_predictions_srl_tmp = 0

    acc_avg = correct_predictions / total_processed_examples
    absa_avg_acc = 0
    if total_processed_predictions_absa != 0:
        absa_avg_acc = correct_predictions_absa / total_processed_predictions_absa
    srl_avg_acc = 0
    if total_processed_predictions_srl != 0:
        srl_avg_acc = correct_predictions_srl / total_processed_predictions_srl

    logger.info(f"Number of examples:{total_processed_examples}")
    logger.info(f"Correct predictions:{correct_predictions}")
    logger.info(f"AVG acc:{acc_avg}")

    logger.info(f"Correct ABSA predictions:{correct_predictions_absa}")
    logger.info(f"Total   ABSA predictions:{total_processed_predictions_absa}")
    logger.info(f"Accuracy ABSA:{absa_avg_acc}")

    logger.info(f"Correct SRL predictions:{correct_predictions_srl}")
    logger.info(f"Total   SRL predictions:{total_processed_predictions_srl}")
    logger.info(f"Accuracy SRL:{srl_avg_acc}")
    logger.info("-------")

    return acc_avg, np.mean(losses)


def get_predictions(model, data_loader, device, batch_size, args, loss_fn, print_progress=False):
    model = model.eval()

    predictions_absa = []
    prediction_probs_absa = []
    real_values_absa = []

    predictions_srl = []
    prediction_probs_srl = []
    real_values_srl = []

    texts_orig = []
    texts_help = []

    losses = []
    correct_predictions = 0
    total_processed_examples = 0
    total_processed_predictions = 0

    total_processed_predictions_absa = 0
    correct_predictions_absa = 0
    total_processed_predictions_srl = 0
    correct_predictions_srl = 0

    if batch_size is None:
        logger.info("Batch size not specified for priting info setting it to 16")
        batch_size = 16

    t0 = time.time()

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if FAST_DEBUG is True:
                # only for testing purposes
                if i == NUM_FAST_DEBUG_ITER:
                    break

            if print_progress is True:
                if i % args.print_stat_frequency == 0:
                    cur_time = time.time() - t0
                    epoch_time = time.time() - epoch_time
                    print("total time:" + str(cur_time) + "s 10 epochs:" + str(epoch_time) + " s  Predicted:" + str(
                        i * batch_size) + " examples current batch:" + str(i))
                    epoch_time = time.time()



            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            if "token_type_ids" in batch:
                token_type_ids = batch['token_type_ids'].to(device)
            else:
                token_type_ids = torch.zeros_like(input_ids).to(device)
            source = batch['source'][0]


            text_orig = ''
            text_help = ''
            try:
                if source == SRL_SOURCE:
                    text_orig = ''
                else:
                    text_orig = batch["text_orig"]
                    text_help = batch["text_help"]
            except Exception:
                logger.info("Failed to add text")


            n_predicted_classes = len(labels.shape)
            # end2end 250 x 250
            if len(labels.shape) == 3:
                n_predicted_classes = labels.shape[1] * labels.shape[2]
            elif len(labels.shape) == 2:
                n_predicted_classes = labels.shape[1]

            # get size of batch
            data_size = len(labels)
            total_processed_examples += data_size
            input = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask, "labels": labels, "return_dict": True}
            if "position_ids" in batch:
                input["position_ids"] = batch["position_ids"].to(device)
            if args.use_custom_model:
                loss, logits = model(**input)
            else:
                # if loss_fn is None then we pass labels, otherwise None and we compute loss by ourselfs
                input["labels"] = labels if loss_fn is None else None
                output_obj = model(**input)

                logits = output_obj.logits
                if loss_fn is None:
                    loss = output_obj.loss
                else:
                    loss = loss_fn(logits, labels)

            if args.data_parallel is True:
                # loss_mean = loss.mean()
                # print("Loss mean:" + str(loss_mean))
                loss = loss.mean()

            losses.append(loss.item())

            _, preds = torch.max(logits, dim=1)


            if source == SRL_SOURCE:
                tmp_labels = labels[labels >= 0]
                tmp_preds = preds[labels >= 0]
                tmp = torch.sum(tmp_preds == tmp_labels)
                tmp = tmp.detach().cpu().int().item()
                total_processed_predictions_srl += len(tmp_labels)
                total_processed_predictions += len(tmp_labels)
                correct_predictions_srl += tmp
            elif source == ABSA_SOURCE:
                tmp = torch.sum(preds == labels)
                tmp = tmp.detach().cpu().int().item()
                total_processed_predictions_absa += data_size
                correct_predictions_absa += tmp
                total_processed_predictions += data_size * n_predicted_classes
            else:
                raise Exception("Unknown source:" + str(source))
            correct_predictions += tmp

            texts_help.extend(text_help)
            texts_orig.extend(text_orig)

            if source == SRL_SOURCE:
                predictions_srl.extend(preds.detach().cpu())
                # prediction_probs_srl.extend(probs.detach().cpu())
                real_values_srl.extend(labels.detach().cpu())

            elif source == ABSA_SOURCE:
                probs = F.softmax(logits, dim=1)
                predictions_absa.extend(preds.detach().cpu())
                prediction_probs_absa.extend(probs.detach().cpu())
                real_values_absa.extend(labels.detach().cpu())
            else:
                raise Exception("Unknown source:" + str(source))

    if len(predictions_absa) > 0:
        predictions_absa = torch.stack(predictions_absa).detach().cpu()
        prediction_probs_absa = torch.stack(prediction_probs_absa).detach().cpu()
        real_values_absa = torch.stack(real_values_absa).detach().cpu()
    if len(predictions_srl) > 0:
        predictions_srl = torch.stack(predictions_srl).detach().cpu()
        # prediction_probs_srl = torch.stack(prediction_probs_srl).detach().cpu()
        real_values_srl = torch.stack(real_values_srl).detach().cpu()

    avg_loss = 0
    if loss_fn is not None:
        avg_loss = np.mean(losses)

    avg_acc = correct_predictions / total_processed_predictions
    absa_avg_acc = 0
    if total_processed_predictions_absa != 0:
        absa_avg_acc = correct_predictions_absa / total_processed_predictions_absa
    srl_avg_acc = 0
    if total_processed_predictions_srl != 0:
        srl_avg_acc = correct_predictions_srl / total_processed_predictions_srl

    logger.info(f"Number of examples:{total_processed_examples}")
    logger.info(f"Number of predictions:{total_processed_predictions}")
    logger.info(f"Correct predictions:{correct_predictions}")
    logger.info(f"AVG acc:{avg_acc}")
    logger.info(f"AVG loss:{avg_loss}")

    logger.info(f"Correct ABSA predictions:{correct_predictions_absa}")
    logger.info(f"Total   ABSA predictions:{total_processed_predictions_absa}")
    logger.info(f"Accuracy ABSA:{absa_avg_acc}")

    logger.info(f"Correct SRL predictions:{correct_predictions_srl}")
    logger.info(f"Total   SRL predictions:{total_processed_predictions_srl}")
    logger.info(f"Accuracy SRL:{srl_avg_acc}")
    logger.info("----")

    return texts_orig, texts_help, predictions_absa, prediction_probs_absa, real_values_absa,\
               avg_loss, avg_acc,\
               predictions_srl, prediction_probs_srl, real_values_srl


