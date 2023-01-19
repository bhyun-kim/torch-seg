import cv2

import os.path as osp 

from glob import glob 
from torch.utils.data import Dataset
from library import DatasetRegistry

"""
References: 

[1] https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

@DatasetRegistry.register('CityscapesDataset')
class CityscapesDataset(Dataset):
    """Cityscapes dataset."""

    def __init__(
        self, root_dir, split='train', transforms=None, classes=None, palette=None, 
        img_suffix = '_leftImg8bit.png', seg_suffix = '_gtFine_labelTrainIds.png'):
        """
        Args: 
            root_dir (str): Directory with all the images.
            split (str): The dataset split, supports 'train', 'val', or 'test'
            classes (tuple)
            palette (list)
            img_suffix (str)
            seg_suffix (str) 

        Folder structure:
            root_dir 
                └leftImg8bit
                    └ train/val/test
                        └ cities 
                            └ ***_leftImg8bit.png
                └gtFine
                    └ train/val/test
                        └ cities
                            └ ***_gtFine_labelTrainIds.png
                
        """

        
        self.root_dir = root_dir
        self.transforms = transforms
        self.split = split

        self.img_list = glob(osp.join(self.root_dir, 'leftImg8bit', self.split, '**', f'*{img_suffix}'), recursive=True)
        self.seg_suffix = seg_suffix
        if classes == None : 
            self.classes = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                'bicycle')
        else :
            self.classes = classes 

        if palette == None:
            self.palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                [0, 80, 100], [0, 0, 230], [119, 11, 32]]
        else:
            self.palette = palette

    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        Args: 
            idx (int): data index

        Returns:
            sample (dict, {'image': np.arr, 'segmap': np.arr})
        """

        img_path = self.img_list[idx]
        segmap_path = img_path.replace('leftImg8bit', 'gtFine')
        segmap_path = segmap_path.replace('.png', self.seg_suffix.replace('_gtFine', ''))
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segmap = cv2.imread(segmap_path, cv2.IMREAD_UNCHANGED)

        sample = {'image': img, 'segmap': segmap} 

        if self.transforms:
            sample = self.transforms(sample)

        return sample