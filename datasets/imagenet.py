
from torchvision import datasets 

from library import DatasetRegistry


@DatasetRegistry.register('ImageNet')
class ImageNet(datasets.ImageNet):

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)

        sample = {'image': img, 'target': target} 

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    