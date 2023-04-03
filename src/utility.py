"""
    CompletionFormer

    ======================================================================

    Some of useful functions are defined here.
"""


import os
import shutil
import torch.optim as optim
import torch.optim.lr_scheduler as lrs


def convert_str_to_num(val, t):
    val = val.replace('\'', '')
    val = val.replace('\"', '')

    if t == 'int':
        val = [int(v) for v in val.split(',')]
    elif t == 'float':
        val = [float(v) for v in val.split(',')]
    else:
        raise NotImplementedError

    return val


def make_optimizer_scheduler(args, target, num_batches=-1):
    # optimizer
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'RMSPROP':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['momentum'] = 0 # default
        kwargs_optimizer['weight_decay'] = 0 # default
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'ADAMW':
        optimizer_class = optim.AdamW
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    else:
        raise NotImplementedError

    trainable = target.parameters()
    optimizer = optimizer_class(trainable, **kwargs_optimizer)
    scheduler = lrs.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    return optimizer, scheduler


def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(
        ".", "..", ".git*", "*pycache*", "*build", "*.fuse*", "*_drive_*",
        "*pretrained*")

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree('.', backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))