from importlib import import_module
from datetime import datetime

import os 
import sys 
import logging


import os.path as osp

import logging
import os
from datetime import datetime


def cvt_pathToModule(file_path):
    """Convert path (string) to module form.

    Args :
        file_path (str) : file path written in nomal path form
    
    Returns :
        module_form (str) : file path in module form (i.e. matplotlib.pyplot)
    
    """
    
    file_path = file_path.replace('/', '.')
    module_form  = file_path.replace('.py', '')
    
    return module_form

def cvt_moduleToDict(mod) :
    """
    Args : 
        mod (module)  
    
    Returns :
        cfg (dict)
    
    """
    cfg = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
        }
    
    return cfg


def cvt_cfgPathToDict(path):
    """Convert configuration path to dictionary to
    Args: 
        path (str)

    Returns: 
        cfg (dict)
    """

    abs_path = osp.abspath(path)

    sys.path.append(osp.split(abs_path)[0])
    _mod = import_module(osp.split(abs_path)[1].replace('.py', ''))

    return cvt_moduleToDict(_mod)


class Logger(object):
    def __init__(self, directory, verbose=1):
        """
        Args:
            directory (str): path to write logging file 
            verbose (int): 
                if verbose == 1: logger print and write on logging files
                if verbose == -1: skips both print and write  
        """

        # build logger  
        self.verbose = verbose

        if self.verbose != -1:

            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            os.makedirs(directory, exist_ok=True)
            
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_file = directory + f'/{current_time}.log'
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.info('Log file is %s' % log_file)

    def info(self, input):
        """
        Args: 
            input (str)
        """
        if self.verbose == 1:
            self.logger.info(input)

        elif self.verbose == -1:
            pass

        return None


