import torch.nn as nn 

from tools.library import EncoderRegistry

class BasicBlock(nn.Module):

    def __init__(
        self,
        in_channel=64,
        out_channel=64,
        downsample=False,
        increase_dim=False,
        bias=False
        ):
        """
        Args:
            in_channel (int): number of input channels
            out_channel (int): number of output channels
            downsample (bool): whether to downsample the input tensor
            increase_dim (bool): whether to increase the dimension of the input tensor 
            bias (bool): whether to use bias in convolution

        """
        super(BasicBlock, self).__init__()

        if downsample:
            stride = 2
        else:
            stride = 1
            

        self.conv0 = nn.Conv2d(
            in_channel, 
            out_channel, 
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias
            )
        self.norm0 = nn.BatchNorm2d(out_channel)
        self.act0 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(
            out_channel, 
            out_channel,
            kernel_size=3,
            padding=1,
            bias=bias
            )
        self.norm1 = nn.BatchNorm2d(out_channel)
        self.act1 = nn.ReLU(inplace=True)

        if increase_dim:
            
            if downsample: 
                self.increase_dim = nn.Sequential(
                    nn.Conv2d(
                        in_channel, 
                        out_channel, 
                        kernel_size=1, 
                        stride=stride,
                        bias=bias,
                        ),
                    nn.BatchNorm2d(out_channel)
                    )
                
            else: 
                self.increase_dim = nn.Sequential(
                    nn.Conv2d(
                        in_channel, 
                        out_channel, 
                        kernel_size=1, 
                        bias=bias
                        ),
                    nn.BatchNorm2d(out_channel)
                    )
                
        else: 
            self.increase_dim = None

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

        if self.increase_dim is not None:
            identity = self.increase_dim(x)  

        out += identity
        out = self.act1(out)

        return out

class Bottleneck(nn.Module):

    def __init__(
        self,
        in_channel=256,
        out_channel=256,
        expansion_rate=4,
        downsample=False,
        increase_dim=False,
        bias=False
        ):
        """

        Args:
            in_channel (int): number of input channels
            out_channel (int): number of output channels
            downsample (bool): whether to downsample the input tensor
            increase_dim (bool): whether to increase the dimension of the input tensor
            bias (bool): whether to use bias in convolution
        """
        super(Bottleneck, self).__init__()

        if downsample:
            stride = 2
        else:
            stride = 1

        self.conv0 = nn.Conv2d(
            in_channel, 
            out_channel // expansion_rate, 
            kernel_size=1,
            bias=bias
            )
        self.norm0 = nn.BatchNorm2d(out_channel // expansion_rate)
        self.act0 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(
            out_channel // expansion_rate, 
            out_channel // expansion_rate,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias
            )
        self.norm1 = nn.BatchNorm2d(out_channel // expansion_rate)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channel // expansion_rate, 
            out_channel, 
            kernel_size=1,
            bias=bias
            )
        self.norm2 = nn.BatchNorm2d(out_channel)
        self.act2 = nn.ReLU(inplace=True)

        if increase_dim:
            
            if downsample: 
                self.increase_dim = nn.Sequential(
                    nn.Conv2d(
                        in_channel, 
                        out_channel, 
                        kernel_size=1, 
                        stride=stride,
                        bias=bias,
                        # padding=1
                        ),
                    nn.BatchNorm2d(out_channel)
                    )
                
            else: 
                self.increase_dim = nn.Sequential(
                    nn.Conv2d(
                        in_channel, 
                        out_channel, 
                        kernel_size=1, 
                        bias=bias
                        ),
                    nn.BatchNorm2d(out_channel)
                    )

        else: 
            self.increase_dim = None

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

        if self.increase_dim is not None:
            identity = self.increase_dim(x)  

        out += identity
        out = self.act2(out)

        return out


@EncoderRegistry.register("resnet")
class ResNet(nn.Module):
    """
    ResNet model

    Default: ResNet-18
    """

    def __init__(
        self,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        channels=[64, 128, 256, 512],
    ):
        """
        Args:
            block (nn.Module): block type
            layers (list): number of layers for each stage
            channels (list): number of channels for each stage
        """
        super(ResNet, self).__init__()

        self.conv0 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
            )

        self.norm0 = nn.BatchNorm2d(64)
        self.act0 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if block == BasicBlock:
            increase_conv2 = False

        elif block == Bottleneck:
            increase_conv2 = True


        self.stage1 = self._make_stage(
            block=block,
            in_channel=64,
            out_channel=channels[0],
            num_layers=layers[0],
            downsample=False,
            increase_dim=increase_conv2
            )

        
        self.stage2 = self._make_stage(
            block=block,
            in_channel=channels[0],
            out_channel=channels[1],
            num_layers=layers[1],
            downsample=True,
            increase_dim=True
            )

        self.stage3 = self._make_stage(
            block=block,
            in_channel=channels[1],
            out_channel=channels[2],
            num_layers=layers[2],
            downsample=True,
            increase_dim=True
            )

        self.stage4 = self._make_stage(
            block=block,
            in_channel=channels[2],
            out_channel=channels[3],
            num_layers=layers[3],
            downsample=True,
            increase_dim=True
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor
        
        Returns:
            out (Tensor): output tensor
        """

        out = self.conv0(x)
        out = self.norm0(out)
        out = self.act0(out)

        out = self.maxpool(out)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        return out

    def _make_stage(
        self,
        block,
        in_channel,
        out_channel,
        num_layers,
        downsample,
        increase_dim
        ):
        """
        Args:
            block (nn.Module): block type
            in_channel (int): number of input channels
            out_channel (int): number of output channels
            num_layers (int): number of layers
            downsample (bool): whether to downsample the input tensor
            increase_dim (bool): whether to increase the dimension of the input tensor
        """
        layers = []

        layers.append(
            block(
                in_channel=in_channel,
                out_channel=out_channel,
                downsample=downsample,
                increase_dim=increase_dim
                )
            )

        for _ in range(num_layers - 1):
            layers.append(
                block(
                    in_channel=out_channel,
                    out_channel=out_channel,
                    downsample=False,
                    increase_dim=False
                    )
                )

        return nn.Sequential(*layers)

@EncoderRegistry.register("resnet18")
class ResNet18(ResNet):
    """
    ResNet-18 model
    """

    def __init__(self):
        super(ResNet18, self).__init__(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            channels=[64, 128, 256, 512]
            )

@EncoderRegistry.register("resnet34")
class ResNet34(ResNet):
    """
    ResNet-34 model
    """

    def __init__(self):
        super(ResNet34, self).__init__(
            block=BasicBlock,
            layers=[3, 4, 6, 3],
            channels=[64, 128, 256, 512]
            )

@EncoderRegistry.register("resnet50")
class ResNet50(ResNet):
    """
    ResNet-50 model
    """

    def __init__(self):
        super(ResNet50, self).__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            channels=[256, 512, 1024, 2048]
            )

@EncoderRegistry.register("resnet101")
class ResNet101(ResNet):
    """
    ResNet-101 model
    """

    def __init__(self):
        super(ResNet101, self).__init__(
            block=Bottleneck,
            layers=[3, 4, 23, 3],
            channels=[256, 512, 1024, 2048]
            )

@EncoderRegistry.register("resnet152")
class ResNet152(ResNet):
    """
    ResNet-152 model
    """

    def __init__(self):
        super(ResNet152, self).__init__(
            block=Bottleneck,
            layers=[3, 8, 36, 3],
            channels=[256, 512, 1024, 2048]
            )




