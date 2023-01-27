# resnet pytorch impelementation 

import torch 
import torch.nn as nn 

class BasicBlock(nn.Module):

    def __init__(
        self,
        in_channels=(64, 64), 
        out_channels=(64, 64),
        kernel_sizes=(3, 3), 
        dilations=(1, 1), 
        strides=(1, 1), 
        biases=(False, False), 
        downsample=None
        ):
        """
        Args:
            in_channels (tuple): number of input channels for each convolution
            out_channels (tuple): number of output channels for each convolution
            kernel_sizes (tuple): kernel size for each convolution
            dilations (tuple): dilation for each convolution
            strides (tuple): stride for each convolution
            biases (tuple): bias for each convolution
            downsample (Module): downsample module

        """

        super(BasicBlock, self).__init__()

        self.downsample = downsample

        self.conv0 = nn.Conv2d(
            in_channels[0], 
            out_channels[0],
            kernel_sizes[0], 
            stride=strides[0],
            dilation=dilations[0],
            bias=biases[0]
            )
        self.norm0 = nn.BatchNorm2d(out_channels[0])
        self.act0 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(
            in_channels[1], 
            out_channels[1],
            kernel_sizes[1], 
            stride=strides[1],
            dilation=dilations[1],
            bias=biases[1]
            )
        self.norm1 = nn.BatchNorm2d(out_channels[1])
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor

        Returns:
            out (Tensor): output tensor

        """

        identity = x

        out = self.conv0(x)
        out = self.norm0(out)
        out = self.act0(out)

        out = self.conv1(x)
        out = self.norm1(out)

        if self.downsample is not None:
            identity = self.downsample(x)  

        out += identity
        out = self.act2(out)

        return out

class BottleneckBlock(nn.Module):

    def __init__(
        self, 
        in_channels=(64, 64, 64), 
        out_channels=(64, 64, 256),
        kernel_sizes=(1, 3, 1), 
        dilations=(1, 1, 1),
        strides=(1, 1, 1), 
        biases=(False, False, False),
        downsample=None
        ):
        """

        Args:
            in_channels (tuple): number of input channels for each convolution
            out_channels (tuple): number of output channels for each convolution
            kernel_sizes (tuple): kernel size for each convolution
            dilations (tuple): dilation for each convolution
            strides (tuple): stride for each convolution
            biases (tuple): bias for each convolution
            downsample (Module): downsample module

        """

        super(BottleneckBlock, self).__init__()

        self.downsample = downsample

        self.conv0 = nn.Conv2d(
            in_channels[0], 
            out_channels[0],
            kernel_sizes[0], 
            stride=strides[0],
            dilation=dilations[0],
            bias=biases[0]
            )
        self.norm0 = nn.BatchNorm2d(out_channels[0])
        self.act0 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(
            in_channels[1], 
            out_channels[1],
            kernel_sizes[1], 
            stride=strides[1],
            dilation=dilations[1],
            bias=biases[1]
            )
        self.norm1 = nn.BatchNorm2d(out_channels[1])
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels[2], 
            out_channels[2],
            kernel_sizes[2], 
            stride=strides[2],
            dilation=dilations[2],
            bias=biases[2]
            )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor
        
        Returns:
            out (Tensor): output tensor
        """

        identity = x

        out = self.conv0(x)
        out = self.norm0(out)
        out = self.act0(out)

        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  

        out += identity
        out = self.act2(out)

        return out


    
class ResNet(nn.Module):
    """
    ResNet model
    """

    def __init__(
        self,
        block=BasicBlock,
        layers=(3, 4, 6, 3),
        channels=(64, 128, 256, 512),
        ):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            )

        self.norm1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block,
            in_channels=(64, 64),
            out_channels=(64, 64),
            kernel_sizes=(3, 3),
            dilations=(1, 1),
            strides=(1, 1),
            biases=(False, False),
            num_blocks=3
            )

        self.layer2 = self._make_layer(
            block,
            in_channels=(64, 64),
            out_channels=(128, 128),
            kernel_sizes=(3, 3),
            dilations=(1, 1),
            strides=(2, 1),
            biases=(False, False),
            num_blocks=4
            )
        
        self.layer3 = self._make_layer(
            block,
            in_channels=(128, 128),
            out_channels=(256, 256),
            kernel_sizes=(3, 3),
            dilations=(1, 1),
            strides=(2, 1),
            biases=(False, False),
            num_blocks=6
            )

        self.layer4 = self._make_layer(
            block,
            in_channels=(256, 256),
            out_channels=(512, 512),
            kernel_sizes=(3, 3),
            dilations=(1, 1),
            strides=(2, 1),
            biases=(False, False),
            num_blocks=3
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(
        self,
        block,
        num_blocks=2,
        kernel_sizes=(3, 3),
        channels=(64, 64),
        dilations=(1, 1),
        strides=(1, 1),
        biases=(False, False),
        downsample=None,
        ):
        """
        """

        layers = []

        for i in range(num_blocks):

            layers.append(
                block(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_sizes=kernel_sizes,
                    dilations=dilations,
                    strides=strides,
                    biases=biases,
                    downsample=downsample
                    )
                )


        return 
       



