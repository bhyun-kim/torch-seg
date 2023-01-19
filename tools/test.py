import sys 
import os
import os.path as osp

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import argparse
import logging 
import json

import torch 
import torchvision as tv 

from torchsummary import summary

from importlib import import_module
from tqdm import tqdm

from utils import * 
from builders.builders import *  

from datetime import datetime

def parse_args():

    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')

    args = parser.parse_args()

    return args


def main():
    from pprint import pprint, pformat
    args = parse_args()

    # build config 
    _cfg = args.config

    abs_path = osp.abspath(_cfg)

    sys.path.append(osp.split(abs_path)[0])
    _mod = import_module(osp.split(abs_path)[1].replace('.py', ''))

    cfg = cvt_moduleToDict(_mod)

    # build logger  
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    os.makedirs(cfg['WORK_DIR'], exist_ok=True)
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_file = cfg['WORK_DIR'] + f'/{current_time}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info('Log file is %s' % log_file)
    
    logger.info(pformat(pprint(cfg, width=100)))

    # select device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### build data loaders 
    data_loaders = build_loaders(cfg['DATA_LOADERS']) 
    # print(len(data_loaders))

    # build loss 
    criterion = build_loss(cfg['LOSS'])
    criterion.to(device)

    # build model     
    model = build_model(cfg['MODEL'])    
    logger.info(summary(model))
    model.to(device)
    PATH = '/home/user/UOS-SSaS Dropbox/05. Data/03. Checkpoints/00. Benchmarks/01. cityscapes/cgnet/11.02.2022/checkpoint_iter_56000.pth'
    model.load_state_dict(torch.load(PATH))

    # build optimizer 
    optimizer = build_optimizer(cfg['OPTIMIZER'], model)
    if  cfg['LR_CONFIG']:
        scheduler = build_lr_config(cfg['LR_CONFIG'], optimizer)

    runner = build_runner(cfg['RUNNER'])
    runner.eval(
        cfg,
        model, 
        device, 
        logger, 
        optimizer, 
        criterion, 
        data_loaders,
        scheduler
    )

    
        
if __name__ == '__main__':
    main()