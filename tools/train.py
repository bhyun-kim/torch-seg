import os
import sys 

import torch 
import logging 
import argparse

import torch.nn as nn
import os.path as osp

from datetime import datetime
from torchsummary import summary
from importlib import import_module
from utils import cvt_cfgPathToDict, Logger

from builders import build_loaders, build_loss, build_model
from builders import build_optimizer, build_lr_config, build_runner

from pprint import pformat


def setup(rank, world_size):
    """Setup training processors 
    Args: 
        rank (int): Processor indentifier
        world_size (int): Number of total processors in the process group 
    """
    os.environ['MASTER_ADDR'] = 'localhost' # TODO check what it means...
    os.environ['MASTER_PORT'] = '12355'     # TODO check what it means...

    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def parse_args():

    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs for training')

    args = parser.parse_args()

    return args


def train(rank):
    """Single GPU training 
    Args: 
        rank (int): Processor indentifier 
    """

    args = parse_args()

    # build config 
    cfg_path = args.config
    num_gpus = args.num_gpus

    if num_gpus > 1: 
        setup(rank, num_gpus)
        is_dist = True
    else: 
        is_dist = None

    cfg = cvt_cfgPathToDict(cfg_path)

    if rank == 0: 
        verbose = 1
    else: 
        verbose = -1

    logger = Logger(cfg['WORK_DIR'], verbose)

    # Print paths 
    pformat_cfg = pformat(cfg, width=75)
    logger.info(pformat_cfg)

    # build data loaders 
    data_loaders = build_loaders(cfg['DATA_LOADERS'], rank, num_replicas=num_gpus) 

    # build model     
    model = build_model(cfg['MODEL'])    
    
    if cfg['LOAD_FROM'] :
        pretrained_dict = torch.load(cfg['LOAD_FROM'])
        pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}

        for k in model.state_dict():
            if k in pretrained_dict: 
                if model.state_dict()[k].shape != pretrained_dict[k].shape: 

                    print(k)
                    pretrained_dict.pop(k)

        # print(model.state_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        print('Load from config is working!')



    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if rank == 0: 
        logger.info(str(summary(model, input_size=(3, cfg['CROP_SIZE'][0], cfg['CROP_SIZE'][1]))))

    if is_dist: 
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    
    # build optimizer 
    optimizer = build_optimizer(cfg['OPTIMIZER'], model)
    scheduler = build_lr_config(cfg['LR_CONFIG'], optimizer)

    runner = build_runner(cfg['RUNNER'])
    runner.train(
        cfg,
        model, 
        device, 
        logger, 
        optimizer, 
        data_loaders,
        scheduler,
        is_dist=is_dist
    )


def dist_train(num_gpus):
    """Spawn Multi-GPU training
    """

    torch.multiprocessing.spawn(
        train,
        nprocs=num_gpus
    )
    

def main():
    
    args = parse_args()

    # select train mode
    if args.num_gpus > 1: 
        dist_train(args.num_gpus)
    else : 
        train(rank=0)

        
if __name__ == '__main__':
    main()