
import torch
import torchvision as tv 

import pipelines 
import datasets
from models import encoders, losses, heads, optims, lr_schedulers, wrappers
import runners

from tools.library import DatasetRegistry, PipelineRegistry, LossRegistry
from tools.library import EncoderRegistry, DecoderRegistry, HeadRegistry
from tools.library import OptimRegistry, SchedulerRegistry, RunnerRegistry

from models.frames.frames import ModelFramer


def build_pipelines(cfg_pipelines):
    """Build pipelines from config

    Args: 
        cfg_pipelines (list): sequence of pipeline configurations

    Returns:
        transform (torchvision.transforms.Compose)
    
    """

    pipelines = [] 

    for cfg in cfg_pipelines: 
        name = cfg.pop('type')
        pipeline = PipelineRegistry.lookup(name)
        pipelines.append(pipeline(**cfg))

    return tv.transforms.Compose(pipelines)



def build_dataset(cfg, transform=None):
    """Build dataset from config 
    Args: 
        cfg (dict): dataset configuration 
        transform (torchvision.transforms)

    Returns: 
        dataset (torch.utils.data.Dataset)
    """
    if cfg['type'] == 'ConcatDataset': 
        dataset = _concat_dataset(cfg, transform=transform)
    else:  
        dataset = _build_dataset(cfg, transform=transform)
    return dataset

def _concat_dataset(cfg, transform=None):
    """Concat datasets in cfg 
    Args: 
        cfg (dict): dataset configuration 
        transform (torchvision.transforms)

    Returns: 
        dataset (torch.utils.data.ConcatDataset)

    """
    datasets = []

    for dataset in cfg['datasets']:
        datasets.append(_build_dataset(dataset, transform=transform))

    return torch.utils.data.ConcatDataset(datasets)

def _build_dataset(cfg, transform=None):
    """Build dataset from config 
    Args: 
        cfg (dict): dataset configuration 

    Returns: 
        dataset (torch.utils.data.Dataset)
    """
    dataset_type = cfg.pop('type')
    cfg['transform'] = transform
    return DatasetRegistry.lookup(dataset_type)(**cfg)


def build_data_loader(cfg, dataset, sampler):
    """Build data loader from config
    Args:
        cfg (dict): data loader configuration 
        dataset (torch.utils.data.Dataset)

    Returns: 
        data_loader (torch.utils.data.DataLoader)
    """
    
    return torch.utils.data.DataLoader(dataset, **cfg, sampler=sampler)

def build_loss(cfg):
    """Build loss from config 
    Args:
        cfg (dict): loss configuration 
    
    Returns: 
        loss (torch.nn.Module)
    """
    loss_type = cfg.pop('type')
    
    return LossRegistry.lookup(loss_type)(**cfg)

def build_model(cfg):
    """Build model from config 
    Args: 
        cfg (dict): model configuration
    
    Returns:
        model (torch.nn.Module)
    """
    
    encoder_name = cfg['encoder'].pop('type')
    encoder = EncoderRegistry.lookup(encoder_name)(**cfg['encoder'])

    head = build_head(cfg['head'])

    if cfg['decoder']:
        decoder_name = cfg['decoder'].pop('type')
        decoder = DecoderRegistry.lookup(decoder_name)( **cfg['decoder'])
    else: 
        decoder = None 

    return ModelFramer(encoder=encoder, decoder=decoder, head=head)


def build_head(cfg):
    """Build head from config
    Args: 
        cfg (dict): optimizer configuration

    Returns: 
        torch.nn.Module
    """

    loss = build_loss(cfg['loss'])
    head_name = cfg.pop('type')
    cfg['loss'] = loss
    return HeadRegistry.lookup(head_name)(**cfg)



def build_optimizer(cfg, model):
    """Build optimizer from config 
    Args: 
        cfg (dict): optimizer configuration
        model (torch.nn.Module)
    
    Returns:
        optimizer (torch.optim)
    """

    optim_type = cfg.pop('type')

    return OptimRegistry.lookup(optim_type)(model.parameters(), **cfg)



def build_loaders(cfg, rank, num_replicas=0):
    """Build data loaders from config 
    Args: 
        cfg (dict): data loaders configuration
        model (torch.nn.Module)
    
    Returns:
        optimizer (torch.optim)
    """
    loaders = dict()
    splits = [k for k, v in cfg.items()]
    for split in splits:
        _cfg = cfg[split]
        
        # build pipelines 
        transform = build_pipelines(_cfg['pipelines'])

        # build dataset 
        dataset = build_dataset(_cfg['dataset'], transform=transform)

        if num_replicas > 0 :
            if split == 'train': 
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset, num_replicas=num_replicas, rank=rank
                    )

                loaders[split] = build_data_loader(_cfg['loader'], dataset, sampler) 

            elif split == 'val':
                if rank == 0:
                    sampler = None 
                    loaders[split] = build_data_loader(_cfg['loader'], dataset, sampler) 
                    
        else : 
            sampler = None 
            loaders[split] = build_data_loader(_cfg['loader'], dataset, sampler) 


    return loaders 


def build_runner(cfg):
    """Build runner from config 
    Args: 
        cfg (dict): runner configuration
    
    Returns:
        runner (obj)
    """

    runner_type = cfg.pop('type')

    return RunnerRegistry.lookup(runner_type)(**cfg)


def build_lr_config(cfg, optim):
    """Build lr_config
    Args: 
        cfg (dict): learning rate config
        optim (torch.optim)
    
    Returns:
        torch.optim.lr_scheduler 
    """

    lr_scheduler_type = cfg.pop('type')

    return SchedulerRegistry.lookup(lr_scheduler_type)(optimizer=optim, **cfg)
