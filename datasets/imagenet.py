

from torchvision import datasets 

from tools.utils import imread
from tools.library import DatasetRegistry


@DatasetRegistry.register('ImageNet')
class ImageNetDataset(datasets.ImageNet):
    def __init__(self, loader=imread, **kwargs):
        super().__init__(loader=loader, **kwargs)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        try : 
            path, target = self.samples[index]
            img = self.loader(path)

            sample = {'image': img, 'target': target} 

            if self.transform is not None:
                sample = self.transform(sample)
        except: 
            return None

        return sample

    