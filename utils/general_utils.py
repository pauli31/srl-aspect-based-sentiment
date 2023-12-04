import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import os
import logging
import json
import torch

from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)

def visaulize_training(history, path):
  plt.plot(history['train_acc'], label='train accuracy')
  plt.plot(history['val_acc'], label='validation accuracy')

  plt.title('Training history')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend()
  plt.ylim([0, 1])

  plt.savefig(path)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def get_actual_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H.%M_%S.%f")


def append_f1_to_file_name(name_file, f1):
    name_file = name_file + "_F1-%.4f" % (f1)
    return name_file


def generate_wandb_group(args):
    time = get_actual_time()

    epochs = args.epoch_num
    batch_size = args.batch_size
    model_name = Path(args.model_name).name
    max_train_data = args.max_train_data
    task = args.task
    if task == 'CAT':
        task = task + "_S-" + str(args.solution_type_cat)
    elif task == 'SRL':
        task = task
    else:
        raise Exception("Unsupported task:" + str(task))

    if max_train_data > 1:
        max_train_data = int(max_train_data)

    dataset_name = args.dataset_name

    name_file = model_name + "_" \
                + dataset_name \
                + "_BS-" + str(batch_size) \
                + "_EC-" + str(epochs) \
                + "_LR-%.7f" % (args.lr) \
                + "_LEN-" + str(args.max_seq_len)

    name_file += "_" + time
    name_file = name_file.replace('.', '-')

    return name_file

def generate_file_name_transformer(args):
    time = get_actual_time()

    epochs = args.epoch_num
    batch_size = args.batch_size
    model_name = Path(args.model_name).name
    max_train_data = args.max_train_data
    task = args.task
    if task == 'CAT':
        task = task + "_S-" + str(args.solution_type_cat)
    elif task == 'SRL':
        task = task
    else:
        raise Exception("Unsupported task:" + str(task))

    if max_train_data > 1:
        max_train_data = int(max_train_data)

    dataset_name = args.dataset_name

    name_file = model_name + "_" \
                + dataset_name \
                + "_T-" + task \
                + "_BS-" + str(batch_size) \
                + "_EC-" + str(epochs) \
                + "_LR-%.7f" % (args.lr) \
                + "_LEN-" + str(args.max_seq_len) \
                + "_SCH-" + str(args.scheduler) \
                + "_TRN-" + str(args.use_only_train_data) \
                + "_MXT-" + str(max_train_data) \
                + "_FRZ-" + str(args.freze_base_model) \
                + "_WD-" + str(args.weight_decay) \
                + "_WRP-" + str(args.warm_up_steps)

    name_file += "_" + time
    name_file = name_file.replace('.', '-')

    return name_file


def save_model_transformer(model, tokenizer, optimizer, args, save_dir, save_as_custom, test_result=None, dev_result=None):
    model_name = Path(args.model_name).name
    name_folder = os.path.join(save_dir, model_name)
    if os.path.exists(name_folder) is not True:
        os.makedirs(name_folder)

    name_file = args.config_name

    if test_result is not None:
        name_file = append_f1_to_file_name(name_file, test_result.f1_micro)

    if test_result is not None:
        name_file = append_f1_to_file_name(name_file, test_result.f1_macro)

    logger.info("File name:" + str(name_file))

    # This is config from command line
    # dump used config
    with open(os.path.join(name_folder, name_file + ".config"), 'w') as outfile:
        json.dump(vars(args), outfile, indent=4)

    print_str_test = "No test results"
    excel_test = "No test excel results"
    print_str_dev = "No dev results"
    excel_dev = "No dev excel results"
    head = "None head"
    # if test_result is not None:
    #     print_str_test = get_print_result("Test_", test_result)
    #     head, excel_test = print_results("Test_", test_result, False, args, False)
    #
    # if dev_result is not None:
    #     print_str_dev = get_print_result("Dev_", dev_result)
    #     _, excel_dev = print_results("Dev_", dev_result, False, args, False)


    with open(os.path.join(name_folder, name_file + ".txt"), 'a') as f:
        f.write(70 * "Test\n")
        f.write(print_str_test)
        f.write('\n')
        f.write(70 * "Dev\n")
        f.write(print_str_dev)
        f.write('\n')
        # print results in excel format
        f.write(head + '\n')
        f.write(excel_test + '\n')
        f.write(excel_dev + '\n')
        f.write('\n')
        f.write("Used Parameters:")
        f.write(build_param_string(vars(args)))

    # https://github.com/huggingface/transformers/issues/7849
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    try:
        if model is not None:
            model_save_dir = os.path.join(name_folder, name_file)
            tokenizer.save_pretrained(model_save_dir)
            if os.path.exists(model_save_dir) is not True:
                os.makedirs(model_save_dir)

            if save_as_custom:
                model_save_path = os.path.join(model_save_dir, 'model_torch.bin')

                # Tohle je recomend verze, ale pak abych to nacetl tak musim mit na vstupu mode
                # save_dict = {'model_state_dict': model.state_dict()}
                # if optimizer is not None:
                #     save_dict['optimizer_state_dict'] = optimizer.state_dict()
                # torch.save(save_dict, model_save_path)

                torch.save(model, model_save_path)
                if optimizer is not None:
                    optimizer_save_path = os.path.join(model_save_dir, 'optimizer_torch.bin')
                    torch.save(optimizer, optimizer_save_path)

            else:
                # save trained model, inspired by http://mccormickml.com/2019/07/22/BERT-fine-tuning/
                model_to_save = model.module if hasattr(model,
                                                        'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(model_save_dir)


            logger.info("Model saved into dir:" + str(model_save_dir))

    except Exception:
        logger.error("Failed to save model", exc_info=True)

    return name_file


def load_custom_model_recomended(model_save_path, model, optimizer=None):
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def load_raw_model(model_save_path, optimizer_save_path=None):
    if optimizer_save_path is not None:
        optimizer = torch.load(optimizer_save_path)
    else:
        optimizer = None

    model = torch.load(model_save_path)

    return model, optimizer


def build_param_string(param_dict):
    string_param = 'Model parameters:\n'
    string_param += '--------------------\n'
    for i, v in param_dict.items():
        if i == 'we_matrix':
            continue
        string_param += str(i) + ": " + str(v) + '\n'

    return string_param