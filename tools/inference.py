

import argparse


import os.path as osp
import sys 

from importlib import import_module

from utils import * 
from builders import build_model, build_pipelines 
from tqdm import tqdm
from glob import glob 

import torch
import cv2 
import numpy as np



def parse_args():

    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('infer_dir', help='the dir to inference images')
    parser.add_argument('--input_ext', help='file extension of input images, default = .jpg',
                        type=str, default='.png')
    parser.add_argument('--output_ext', help='file extension of output images, default = .png',
                        type=str, default='.jpg')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # parse args
    cfg_path = args.config
    infer_dir = args.infer_dir
    checkpoint = args.checkpoint
    input_ext = args.input_ext
    output_ext = args.output_ext

    checkpoint = osp.abspath(checkpoint)
    cfg = cvt_cfgPathToDict(cfg_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    model = build_model(cfg['MODEL'])

    
    pretrained_dict = torch.load(checkpoint)
    
    # When model trained with nn.DataParallel, the model is wrapped with 'module'.
    # Remove 'module' from dict for normal inference mode. 
    # See https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/10
    pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
    model.load_state_dict(pretrained_dict)
    model.to(device)

    transforms = build_pipelines(cfg['TEST_PIPELINES'])

    infer_list = glob(osp.join(infer_dir, f'*{input_ext}'))

    for img_path in tqdm(infer_list, desc='Inference'):
        _img = cv2.imread(img_path)
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        dummy_segmap = np.zeros(_img.shape[:2], dtype=np.uint8) # dummy label for usage of pipeliens 
        input = transforms({'image': _img, 'segmap': dummy_segmap})

        img = input['image'].to(device)
        img = torch.unsqueeze(img, dim=0)

        output = model(img)
        output = torch.argmax(output, dim=1)
        output = torch.squeeze(output)

        output = output.detach().cpu().numpy()
        
        # resize output 
        output = cv2.resize(output, (_img.shape[1], _img.shape[0]), interpolation=cv2.INTER_NEAREST) 

        # put_color
        for idx, color in enumerate(cfg['PALETTE']):
            _img[output == idx] = _img[output == idx] * 0.5 + np.array(color)*0.5
        save_path = img_path.replace(input_ext, f'_result{output_ext}')
        
        _img = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, _img)



        
if __name__ == '__main__':
    main()