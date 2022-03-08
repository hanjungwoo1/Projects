from model.unet import unet_4156
from model.basic_cnn import CNN
from utils.test_utils import tester

import argparse
import logging
import os


def parse_argument():
    parser = argparse.ArgumentParser(description = 'test for A2C/A4C data')
    parser.add_argument('--model', type=str, default='Unet', help='Choose model to test [defalut : U-Net]')
    parser.add_argument('--batch_size', type=int, default = None, help='Test batch size [default : 2]')
    parser.add_argument('--checkpoint_dir', type=str, default='log', help='Choose directory to load the checkpoint [default:log]')
    parser.add_argument('--task', type=str, default='A2C', help='Dataset Task[A2C / A4C]')
    parser.add_argument('--num_vote', type=int, default = None, help='Number of voting [default : 100]')
    args = parser.parse_args()

    return args

def logger_initialize(args):
    log_dir = args.checkpoint_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_fname = os.path.join(log_dir, 'test_log_{:s}_{:s}.txt'.format(args.task, args.model))
    LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    DATE_FORMAT = '%Y%m%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
    logger = logging.getLogger(__name__)

    return logger

if __name__ == '__main__' :
    args = parse_argument()
    if args.model == 'CNN' :
        model = CNN()
    else :
        model = unet_4156()
    logger = logger_initialize(args)
    testing = tester(args.task, model, logger, args)
    testing.test()