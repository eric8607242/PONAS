import os
import sys
import logging


class AverageMeter(object):
    def __init__(self, name=''):
        self._name = name
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val, n=1):
        self.sum += val*n
        self.cnt += n
        self.avg = self.sum/self.cnt

    def __str__(self):
        return "%s: %.4f" % (self._name, self.avg)

    def get_avg(self):
        return self.avg

    def __repr__(self):
        return self.__str__()


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0/batch_size))

    return res


def get_logger(log_dir=None):
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    log_format = "%(asctime)s | %(message)s"

    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format=log_format,
                        datefmt="%m/%d %I:%M:%S %p")

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "logger"))
        file_handler.setFormatter(logging.Formatter(log_format))

    logging.getLogger().addHandler(file_handler)
