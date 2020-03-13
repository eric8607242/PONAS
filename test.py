import time
import logging
import argparse

import torch
import torch.nn as nn

from collections import OrderedDict

from models.PONAS_A import PONASA
from models.PONAS_B import PONASB
from models.PONAS_C import PONASC

from utils.countmacs import MAC_Counter
from utils.evaluation import calculate_latency, calculate_param_nums
from utils.util import get_logger, AverageMeter, accuracy
from utils.config import get_config
from utils.dataflow import get_dataset, get_dataloader, get_transforms


def val(model, loader, device):
    top1 = AverageMeter()

    model.eval()
    start_time = time.time()

    with torch.no_grad():
        for step, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)
            N = X.shape[0]

            outs = model(X)

            prec1 = accuracy(outs, y, topk=(1,))[0]
            top1.update(prec1.item(), N)
        top1_avg = top1.get_avg()
        logging.info("Test: Final Prec@1 {:.2%} Time {:.2f}".format(top1_avg, time.time()-start_time))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="path to the config file", required=True)
    parser.add_argument("--model", type=str, help="model", required=True)
    parser.add_argument("--pretrain", type=str, help="path to the pretrain weight", required=True)
    args = parser.parse_args()

    CONFIG = get_config(args.cfg)

    device = torch.device("cuda") if torch.cuda.is_available() and CONFIG.cuda else torch.device("cpu")

    logger = get_logger(CONFIG.log_dir)

    _, _, test_transform = get_transforms(CONFIG)
    _, _, test_dataset = get_dataset(None, None, test_transform, CONFIG)
    _, _, test_loader = get_dataloader(None, None, test_dataset, CONFIG)

    if args.model == "PONASA":
        model = PONASA()
    elif args.model == "PONASB":
        model = PONASB()
    elif args.model == "PONASC":
        model = PONASC()
    model.load_state_dict(torch.load(args.pretrain))

    latency = calculate_latency(model, 3, CONFIG.input_size)
    counter = MAC_Counter(model, [1, 3, CONFIG.input_size, CONFIG.input_size])
    macs = counter.print_summary(False)
    param_nums = calculate_param_nums(model)

    logging.info("Inference time : {:.5f}".format(latency))
    logging.info("MACs number(M) : {}".format(macs["total_gmacs"]*1000))
    logging.info("Parameter numbers : {}".format(param_nums))

    model = model.to(device)
    if device.type == "cuda" and CONFIG.ngpu > 1:
        logging.info("Multiple GPUs loading: {}".format(CONFIG.ngpu))
        model = nn.DataParallel(model, list(range(CONFIG.ngpu)))

    val(model, test_loader, device)
