from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_polynomial_decay_schedule_with_warmup, get_constant_schedule, AdamW, get_constant_schedule_with_warmup

from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT
import torch
import logging
import math



logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)

def get_optimizer(model_params, args):
    """
    # Learning rate decay, can be implemented with tf.keras.optimizers.schedules.LearningRateSchedule


    :param optimizer_name:
    :param lr:  floating point value, or a schedule that is a
    :return:
    """
    lr = args.lr
    optimizer_name = args.optimizer

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model_params, lr=lr)

    elif optimizer_name == 'AdamW':
        optimizer = AdamW(model_params, lr=lr, weight_decay=args.weight_decay)

    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model_params, lr=lr)

    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(model_params, lr=lr)

    elif optimizer_name == 'Adadelta':
        optimizer = torch.optim.Adadelta(model_params, lr=lr, rho=args.adadelta_rho)

    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model_params, lr=lr)
    else:
        raise Exception('Not valid optimizer: ' + optimizer_name)

    return optimizer


def get_lr_scheduler(args, optimizer, total_steps):
    warm_up_steps = args.warm_up_steps
    scheduler_name = args.scheduler

    if warm_up_steps > 0:
        if warm_up_steps == 1:
            raise Exception("Warmup steps cannot be 1")
        if warm_up_steps < 1:
            warm_up_steps = warm_up_steps * total_steps
            warm_up_steps = math.ceil(warm_up_steps)

    logger.info("Number of warm up steps: " + str(warm_up_steps) + " out of: " + str(
        total_steps) + " original warmup steps: " + str(warm_up_steps))

    # https://huggingface.co/transformers/main_classes/optimizer_schedules.html#learning-rate-schedules-pytorch
    if scheduler_name == 'linear_wrp':
        # linearly decreasing
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=total_steps
        )
    elif scheduler_name == 'cosine_wrp':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=total_steps
        )
    elif scheduler_name == 'polynomial_wrp':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=total_steps,
            power=2
        )
    elif scheduler_name == 'constant':
        scheduler = get_constant_schedule(optimizer)
    elif scheduler_name == 'constant_warmp_up':
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up_steps,
        )

    else:
        raise Exception(f"Unknown scheduler: {args.scheduler}")

    return scheduler