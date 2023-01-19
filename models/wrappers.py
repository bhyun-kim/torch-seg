import cv2 
import numpy as np 

from library import WrapperRegistry

@WrapperRegistry.register('Erode')
class Erode(object): 
    """Erode Label
    """

    def __init__(self, target_class, kernel_size=3, iteration=1):
        """
        Args:
            target_class (int): Target class to erode label 
            kernel_size (int or tuple): kernel_size for erosion
            iteration (int): num of iteration to run erosion
        """

        assert isinstance(target_class, int)
        assert isinstance(kernel_size, (int, tuple))
        assert isinstance(iteration, int)

        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2


        self.target_class = target_class
        self.kernel_size = kernel_size
        self.iteration = iteration


    def __call__(self, segmap):
        """
        Args:
            segmap: np.arr (H x W,  uint8)
        
        Returns:
            segmap: np.arr (H x W, uint8)
        """

        segmap_erosion = segmap == self.target_class 

        
        if isinstance(self.kernel_size, tuple):
            kernel = np.ones(self.kernel_size, np.uint8)
        elif isinstance(self.kernel_size, int):
            kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        segmap_erosion = cv2.erode(segmap_erosion, kernel, iteration=self.iteration)

        inpaint_mask = segmap == self.target_class 
        segmap = cv2.inpaint(segmap, inpaint_mask, 0, cv2.INPAINT_NS)
        
        segmap[segmap_erosion] = self.target_class

        return segmap