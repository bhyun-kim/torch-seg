import torch.nn as nn 

from tools.library import HeadRegistry

@HeadRegistry.register('Classify')
class Classify(nn.Module):
    def __init__(
        self,
        loss,
        num_classes,
        in_channels,
        ):

        super().__init__()
        self.criterion = loss
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.fc = nn.Linear(in_channels, num_classes)


    def forward(self, input, labels):
        """
        Args:
            input (Tensor): input tensor
        
        Returns:
            output (Tensor): output tensor
        """
        output = self.predict(input)
        return self.criterion(output, labels)

    def predict(self, input):
        """
        Args:
            input (Tensor): input tensor

        Returns:
            output (Tensor): output tensor
        """
        return self.fc(input)

