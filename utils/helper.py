import torch
import logging
from collections import OrderedDict
import os


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def one_hot(label, depth=10):
    """
    One-hot encode
    """
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)

    return out.long()


def resume_checkpoint(model, checkpoint_path):
    optimizer_state = None
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            if 'optimizer' in checkpoint:
                optimizer_state = checkpoint['optimizer']
            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save
            logging.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            logging.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return optimizer_state, resume_epoch
    else:
        logging.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()
