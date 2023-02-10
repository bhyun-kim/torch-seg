import random

import cv2 
import torch

import numpy as np 

from tools.library import PipelineRegistry

from copy import deepcopy

"""
References: 
[1] https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
[2] https://mmcv-jm.readthedocs.io/en/stable/_modules/mmcv/image/normalize.html
"""

@PipelineRegistry.register('Rescale')
class Rescale(object): 
    """Rescale the image in a sample to a given size.
    """

    def __init__(self, output_size):
        """
        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size(H x W). If int, smaller of image edges is matched
                to output_size keeping aspect ratio the same.
        """

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, tuple):
            assert len(output_size) == 2
        self.output_size = output_size
        self.target_keys = ['image', 'segmap']


    def __call__(self, sample):
        """
        Args:
            sample (dict) 
        
        Returns:
            sample (dict)
        """

        for k in sample.keys():

            if k in self.target_keys:
                if k == 'image':
                    sample[k] = self._rescale(sample[k])
                elif k == 'segmap':
                    sample[k] = self._rescale(
                        sample[k],
                        interpolation=cv2.INTER_NEAREST
                        )

        return sample

    def _rescale(self, array, interpolation=cv2.INTER_LINEAR):
        """
        Args:
            array (np.ndarray): image or label array
        Returns: 
            array (np.ndarray): rescaled image or label array
        """
        h, w = array.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        return cv2.resize(array, (new_w, new_h), interpolation=interpolation)

@PipelineRegistry.register('Dilate')
class Dilate(object): 
    """Dilate Label
    """

    def __init__(self, target_class, kernel_size, iteration=1):
        """
        Args:
            target_class (int): Target class to dilate label 
            kernel_size (int or tuple): kernel_size for dilation 
            iteration (int): num of iteration to run dilation
        """

        assert isinstance(target_class, int)
        assert isinstance(kernel_size, (int, tuple))
        assert isinstance(iteration, int)

        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2

        self.target_class = target_class
        self.kernel_size = kernel_size
        self.iteration = iteration
        self.target_keys = ['segmap']


    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """

        for k in sample.keys():

            if k in self.target_keys:
                sample[k] = self._dilate(sample[k])

    def _dilate(self, segmap):
        """
        Args:
            segmap (np.arr, (H x W, uint8)): Segmentation map
        Returns:
            segmap (np.arr, (H x W, uint8)): Segmentation map
        """

        segmap_dilate = segmap == self.target_class 

        
        if isinstance(self.kernel_size, tuple):
            kernel = np.ones(self.kernel_size, np.uint8)
        elif isinstance(self.kernel_size, int):
            kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        segmap_dilate = cv2.dilate(segmap_dilate, kernel, iteration=self.iteration)

        segmap[segmap_dilate] = self.target_class

        return segmap


@PipelineRegistry.register('RandomRescale')
class RandomRescale(object): 
    """Rescale the image to a randomly selected output size within the given range.
    """

    def __init__(self, output_range):
        """
        Args:
            output_range (tuple, (min, max)): Desired output range. 
        """

        assert isinstance(output_range, (tuple))
        self.output_range = output_range


    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """
        output_size = random.randint(self.output_range[0], self.output_range[1])
        rescale = Rescale(output_size)
        sample = rescale(sample)

        return sample 


@PipelineRegistry.register('Pad')
class Pad(object):
    """Pad image.
    """

    def __init__(self, pad_size, ignore_label=255):
        """
        Args:
            pad_size (tuple or int): Desired pad_size.
        """
        assert isinstance(pad_size, (int, tuple))
        if isinstance(pad_size, int):
            self.pad_size = (pad_size)
        else:
            assert len(pad_size) in [2, 4]
            self.pad_size = pad_size

        self.ignore_label = ignore_label
        self.target_keys = ['image', 'segmap']

    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        top: It is the border width in number of pixels in top direction. 
        bottom: It is the border width in number of pixels in bottom direction. 
        left: It is the border width in number of pixels in left direction. 
        right: It is the border width in number of pixels in right direction. 
        """

        for k in sample.keys():

            if k in self.target_keys:

                if k == 'image':
                    pad_values = (0, 0, 0)
                elif k == 'segmap':
                    pad_values = (self.ignore_label, )

                sample[k] = self._pad(sample[k], pad_values)

        return sample


    def _pad(self, array, pad_values):
        """
        Args: 
            array (np.arr): Array to be padded.
        
        Returns:
            array (np.arr): Padded array.
        """

        if len(self.pad_size) == 1: 
            pad_t, pad_b, pad_l, pad_r = (self.pad_size, )*4 
        elif len(self.pad_size) == 2: 
            pad_t, pad_b = self.pad_size[0], self.pad_size[0]
            pad_l, pad_r = self.pad_size[1], self.pad_size[1]
        elif len(self.pad_size) == 4: 
            pad_t, pad_b, pad_l, pad_r = self.pad_size

        return cv2.copyMakeBorder(
            array, pad_t, pad_b, pad_l, pad_r, 
            cv2.BORDER_CONSTANT, value=pad_values
            )


@PipelineRegistry.register('Crop')
class Crop(object):
    """
    Crop image.
    """

    def __init__(self, top, left, height, width):
        """
        Args: 
            top (int): Top coordinate.
            left (int): Left coordinate.
            height (int): Height of the crop.
            width (int): Width of the crop.
        """
        assert isinstance(top, int)
        assert isinstance(left, int)
        assert isinstance(height, int)
        assert isinstance(width, int)

        self.top = top
        self.left = left
        self.height = height
        self.width = width

        self.target_keys = ['image', 'segmap']


    def __call__(self, sample):

        for k in sample.keys():

            if k in self.target_keys:

                sample[k] = self._crop(sample[k])

        return sample

    def _crop(self, array):
        """
        Args: 
            array (np.arr): Array to be cropped.
        
        Returns:
            array (np.arr): Cropped array.
        """

        return array[self.top:self.top+self.height, self.left:self.left+self.width]



@PipelineRegistry.register('RandomCrop')
class RandomCrop(object):
    """Crop randomly the image in a sample.
    """

    def __init__(self, output_size, cat_max_ratio=0.75, ignore_idx=255.):
        """
        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size(H x W). If int, square crop is made.
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_idx = ignore_idx
        self.target_keys = ['image', 'segmap']

    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """

        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        h_margin, w_margin = h - new_h, w - new_w
        pad_h, pad_w = -h_margin, -w_margin

        if (pad_h >= 0) or (pad_w >= 0): 
            
            pad_h = int(np.ceil(max(pad_h + 1, 0)/2))
            pad_w = int(np.ceil(max(pad_w + 1, 0)/2))
            
            pad = Pad((pad_h, pad_w))
            sample = pad(sample)

            image = sample['image']

            h, w = image.shape[:2]
            new_h, new_w = self.output_size

            h_margin, w_margin = h - new_h, w - new_w

        for i in range(10):

            top = np.random.randint(0, h_margin)
            left = np.random.randint(0, w_margin)

            crop = Crop(top, left, new_h, new_w)
            _sample = deepcopy(sample)
            _sample = crop(_sample)

            # check sample has a key 'segmap'
            if 'segmap' in sample.keys(): 

                uniques, cnt = np.unique(_sample['segmap'], return_counts=True)
                cnt = cnt[uniques != self.ignore_idx]

                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break
            else: 
                break

        return _sample


@PipelineRegistry.register('RandomFlipLR')
class RandomFlipLR(object):
    """
    Horizontally flip the image and segmap.
    """

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): The flipping probability. Between 0 and 1.
        """
        self.prob = prob
        assert prob >=0 and prob <= 1

        self.target_keys = ['image', 'segmap']

    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """

        if np.random.rand() < self.prob:
        
            for k in sample.keys():

                if k in self.target_keys:
                    sample[k] = cv2.flip(sample[k], 1)

        return sample



@PipelineRegistry.register('RandomFlipUD')
class RandomFlipUD(object):
    """
    Vertically flip the image and segmap.
    """

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): The flipping probability. Between 0 and 1.
        """
        self.prob = prob
        assert prob >=0 and prob <= 1
        self.target_keys = ['image', 'segmap']

    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """

        if np.random.rand() < self.prob:
        
            for k in sample.keys():

                if k in self.target_keys:
                    sample[k] = cv2.flip(sample[k], 0)

        return sample

@PipelineRegistry.register('Normalization')
class Normalization(object):
    """Normalize image 
    """
    def __init__(self, mean, std):
        """
        Args:
            mean (tuple, list): (R, G, B)
            std (tuple, list): (R, G, B)
        """
        assert isinstance(mean, (tuple, list))
        assert isinstance(std, (tuple, list))
        assert len(mean) == 3
        assert len(std) == 3

        mean, std = np.array(mean), np.array(std)
        self.mean = np.float64(mean.reshape(1, -1))
        self.stdinv = 1 / np.float64(std.reshape(1, -1))

        self.target_keys = ['image']
        

    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, float32), segmap: np.arr (H x W, uint8)})
        """

        for k in sample.keys():

            if k in self.target_keys:
                sample[k] = self._normalize(sample[k])

        return sample

    def _normalize(self, array):

        array = np.float32(array) if array.dtype != np.float32 else array.copy()

        cv2.subtract(array, self.mean, array)
        cv2.multiply(array, self.stdinv, array)

        return array

@PipelineRegistry.register('ImgToTensor')
class ImgToTensor(object):
    """Convert image arrays in sample to Tensors."""

    def __init__(self):

        self.target_keys = ['image']

    
    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C), segmap: np.arr (H x W)})
        
        Returns:
            sample (dict, {image: torch.tensor (C x H x W), segmap: torch.tensor (H x W)})
        """

        for k in sample.keys():

            if k in self.target_keys:
                sample[k] = self._to_tensor(sample[k])

        return sample

    def _to_tensor(self, array):

        """
        Args:
            array (np.arr, H x W x C)
        Returns:
            array (torch.tensor, C x H x W)
        """

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        array = np.float32(array) if array.dtype != np.float32 else array.copy()
        array = array.transpose((2, 0, 1)) 

        # array to torch.tensor
        return torch.from_numpy(array)
 

@PipelineRegistry.register('SegToTensor')
class SegToTensor(object):
    """Convert segmenatation map in sample to Tensors."""


    def __init__(self):

        self.target_keys = ['segmap']

    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C), segmap: np.arr (H x W)})
        
        Returns:
            sample (dict, {image: torch.tensor (C x H x W), segmap: torch.tensor (H x W)})
        """
            
        for k in sample.keys():

            if k in self.target_keys:
                sample[k] = self._to_tensor(sample[k])

        return sample


    def _to_tensor(self, array):            
        """
        Args:
            array (np.arr, H x W)
        Returns:
            array (torch.tensor, H x W)
        """

        array = np.int64(array) if array.dtype != np.int64 else array.copy()
        array = torch.from_numpy(array)

        return array