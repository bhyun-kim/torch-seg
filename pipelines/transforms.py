import random

import cv2 
import torch

import numpy as np 

from library import PipelineRegistry

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


    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """
        image, segmap = sample['image'], sample['segmap']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        segmap = cv2.resize(segmap, (new_w, new_h),  interpolation = cv2.INTER_NEAREST)

        return {'image': img, 'segmap': segmap}


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


    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """

        image, segmap = sample['image'], sample['segmap']

        segmap_dilate = segmap == self.target_class 

        
        if isinstance(self.kernel_size, tuple):
            kernel = np.ones(self.kernel_size, np.uint8)
        elif isinstance(self.kernel_size, int):
            kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)


        segmap_dilate = cv2.dilate(segmap_dilate, kernel, iteration=self.iteration)

        segmap[segmap_dilate] = self.target_class

        return {'image': image, 'segmap': segmap}


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
        return rescale(sample)


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

    def __call__(self, sample):
        """
        top: It is the border width in number of pixels in top direction. 
        bottom: It is the border width in number of pixels in bottom direction. 
        left: It is the border width in number of pixels in left direction. 
        right: It is the border width in number of pixels in right direction. 
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """
        image, segmap = sample['image'], sample['segmap']

        if len(self.pad_size) == 1: 
            pad_t, pad_b, pad_l, pad_r = (self.pad_size, )*4 
        elif len(self.pad_size) == 2: 
            pad_t, pad_b = self.pad_size[0], self.pad_size[0]
            pad_l, pad_r = self.pad_size[1], self.pad_size[1]
        elif len(self.pad_size) == 4: 
            pad_t, pad_b, pad_l, pad_r = self.pad_size

        image = cv2.copyMakeBorder(
            image, pad_t, pad_b, pad_l, pad_r, 
            cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0)
            )
        segmap = cv2.copyMakeBorder(
            segmap, pad_t, pad_b, pad_l, pad_r, 
            cv2.BORDER_CONSTANT, value=(self.ignore_label,))

        return {'image': image, 'segmap': segmap}


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

    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """
        image, segmap = sample['image'], sample['segmap']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        h_margin, w_margin = h - new_h, w - new_w
        pad_h, pad_w = -h_margin, -w_margin
    
        
        if (pad_h >= 0) or (pad_w >= 0): 
            
            pad_h = int(np.ceil(max(pad_h + 1, 0)/2))
            pad_w = int(np.ceil(max(pad_w + 1, 0)/2))
            
            pad = Pad((pad_h, pad_w))
            sample = pad(sample)

            image, segmap = sample['image'], sample['segmap']

            h, w = image.shape[:2]
            new_h, new_w = self.output_size

            h_margin, w_margin = h - new_h, w - new_w


        for i in range(10):
            
            top = np.random.randint(0, h_margin)
            left = np.random.randint(0, w_margin)

            crop_image = image[top: top + new_h,
                        left: left + new_w]

            crop_segmap = segmap[top: top + new_h,
                            left: left + new_w]

            uniques, cnt = np.unique(crop_segmap, return_counts=True)
            cnt = cnt[uniques != self.ignore_idx]

            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                break
        
        return {'image': crop_image, 'segmap': crop_segmap}


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

    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """
        image, segmap = sample['image'], sample['segmap']

        if np.random.rand() < self.prob:
            # flip image
            image = cv2.flip(image, 1)
            # flip segmap
            segmap = cv2.flip(segmap, 1)

        return {'image': image, 'segmap': segmap}


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

    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """
        image, segmap = sample['image'], sample['segmap']

        if np.random.rand() < self.prob:
            # flip image
            image = cv2.flip(image, 0)
            # flip segmap
            segmap = cv2.flip(segmap, 0)

        return {'image': image, 'segmap': segmap}

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
        

    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, float32), segmap: np.arr (H x W, uint8)})
        """
        image, segmap = sample['image'], sample['segmap']
        image = np.float32(image) if image.dtype != np.float32 else image.copy()

        cv2.subtract(image, self.mean, image)
        cv2.multiply(image, self.stdinv, image)

        return {'image': image, 'segmap': segmap}

@PipelineRegistry.register('ToTensor')
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C), segmap: np.arr (H x W)})
        
        Returns:
            sample (dict, {image: torch.tensor (C x H x W), segmap: torch.tensor (H x W)})
        """
        image, segmap = sample['image'], sample['segmap']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = np.float32(image) if image.dtype != np.float32 else image.copy()
        segmap = np.int64(segmap) if segmap.dtype != np.int64 else segmap.copy()
        
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'segmap': torch.from_numpy(segmap)}
