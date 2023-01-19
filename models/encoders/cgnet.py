import torch 
import torch.nn as nn 

from library import EncoderRegistry

"""
References: 
    [1] https://github.com/wutianyiRosun/CGNet/blob/master/model/CGNet.py
    [2] https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/backbones/cgnet.py
"""
class ConvBNPReLU(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BNPReLU(nn.Module):
    def __init__(self, nOut):
        """
        args:
           nOut: channels of output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output

class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        return output

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output

class ChannelWiseConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        Args:
            nIn: number of input channels
            nOut: number of output channels, default (nIn == nOut)
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output

class DilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output

class ChannelWiseDilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, default (nIn == nOut)
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups= nIn, bias=False, dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output

class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """
    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ContextGuidedBlock_Down(nn.Module):
    """
    the size of feature map divided 2, (H,W,C)---->(H/2, W/2, 2C)
    """
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
        """
        args:
           nIn: the channel of input feature map
           nOut: the channel of output feature map, and nOut=2*nIn
        """
        super().__init__()
        self.conv1x1 = ConvBNPReLU(nIn, nOut, 3, 2)  #  size/2, channel: nIn--->nOut
        
        self.F_loc = ChannelWiseConv(nOut, nOut, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate)
        
        self.bn = nn.BatchNorm2d(2*nOut, eps=1e-3)
        self.act = nn.PReLU(2*nOut)
        self.reduce = Conv(2*nOut, nOut,1, 1)  #reduce dimension: 2*nOut--->nOut
        
        self.F_glo = FGlo(nOut, reduction)    

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur], 1)  #  the joint feature
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)     #channel= nOut
        
        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature

        return output


class ContextGuidedBlock(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, 
           add: if true, residual learning
        """
        super().__init__()
        n= int(nOut/2)
        self.conv1x1 = ConvBNPReLU(nIn, n, 1, 1)  #1x1 Conv is employed to reduce the computation
        self.F_loc = ChannelWiseConv(n, n, 3, 1) # local feature
        self.F_sur = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate) # surrounding context
        self.bn_prelu = BNPReLU(nOut)
        self.add = add
        self.F_glo= FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        
        joi_feat = torch.cat([loc, sur], 1) 

        joi_feat = self.bn_prelu(joi_feat)

        output = self.F_glo(joi_feat)  #F_glo is employed to refine the joint feature
        # if residual version
        if self.add:
            output  = input + output
        return output

class InputInjection(nn.Module):
    def __init__(self, downsamplingRatio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, downsamplingRatio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))
    def forward(self, input):
        for pool in self.pool:
            input = pool(input)
        return input


@EncoderRegistry.register('CGNet')
class CGNet(nn.Module):
    def __init__(self, 
                 classes=19, 
                 channels=(32, 64, 128),
                 blocks=(3, 21),
                 dilations=(2, 4),
                 reductions=(8, 16),
                 dropout=False
                 ):
        """
        Args:
          classes (int): number of classes in the dataset. Default is 19 for the cityscapes
          channels (tuple): number of channels in stage 1-3
          blocks (tuple): number of blocks in stage 2-3 
          dilations (tuple): dilation rates of conv layers in stage 2-3 
          reductions (tuple): feature map size reduction of global extractor in stage 1-2
          dropout (bool): if true, use dropout layer before the last conv layer 

        Refs:
            [1] Wu, T., Tang, S., Zhang, R., Cao, J., & Zhang, Y. (2020). 
                Cgnet: A light-weight context guided network for semantic segmentation. 
                IEEE Transactions on Image Processing, 30, 1169-1179.
        """
        super().__init__()

        # input injection 
        self.inject1 = InputInjection(int(reductions[0]/8))  #down-sample for Input Injection, factor=2
        self.inject2 = InputInjection(int(reductions[1]/8))  #down-sample for Input Injiection, factor=4

        # stage 1 
        self.stage1 = nn.ModuleList()
        self.stage1.append(ConvBNPReLU(3, channels[0], 3, 2))  # feature map size divided 2, 1/2)  
        self.stage1.append(ConvBNPReLU(channels[0], channels[0], 3, 1))                          
        self.stage1.append(ConvBNPReLU(channels[0], channels[0], 3, 1))   
        self.bn1 = BNPReLU(channels[0]+3)  # the input injection output channel is 3
        
        #stage 2
        self.stage2 = nn.ModuleList()
        self.stage2.append(
            ContextGuidedBlock_Down(
                channels[0]+3,
                channels[1], 
                dilation_rate=dilations[0],
                reduction=reductions[0]
            )  
        )

        for _ in range(0, blocks[0]-1):
            self.stage2.append(
                ContextGuidedBlock(
                    channels[1], 
                    channels[1],
                    dilation_rate=dilations[0],
                    reduction=reductions[0]
                )
            )  

        self.bn2 = BNPReLU(channels[2]+3) # the input injection output channel is 3
        
        #stage 3
        self.stage3 = nn.ModuleList()
        self.stage3.append(
            ContextGuidedBlock_Down(
                channels[2]+3,
                channels[2],
                dilation_rate=dilations[1],
                reduction=reductions[1]
            ) 
        )

        for _ in range(0, blocks[1]-1):
            self.stage3.append(
                ContextGuidedBlock(
                    channels[2], 
                    channels[2],
                    dilation_rate=dilations[1],
                    reduction=reductions[1]
                )
            )  
        
        self.bn3 = BNPReLU(channels[1]+channels[1]+channels[2])

        if dropout:
            self.classifier = nn.Sequential(nn.Dropout2d(0.1, False),Conv(256, classes, 1, 1))
        else:
            self.classifier = nn.Sequential(Conv(256, classes, 1, 1))

        #init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname in ['Conv2d', 'Linear']:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

            elif classname in ['BatchNorm2d']:
                nn.init.constant_(m.weight, val=1)

            elif classname in ['PReLU']:
                nn.init.constant_(m.weight, val=0)
                

    def forward(self, input):
        """
        Args:
            input (torch.tensor): Receives the input RGB image

        Returns 
            (torch.tensor): segmentation map
        """
        
        # input injection 
        inp1 = self.inject1(input)
        inp2 = self.inject2(input)
        
        # stage 1
        for i, layer in enumerate(self.stage1):
            if i==0:
                output1 = layer(input)
            else:
                output1 = layer(output1)

        output1_cat = self.bn1(torch.cat([output1, inp1], 1))

        # stage 2
        for i, layer in enumerate(self.stage2):
            if i==0:
                output2_0 = layer(output1_cat)
            elif i==1:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.bn2(torch.cat([output2, output2_0, inp2], 1))

        # stage 3
        for i, layer in enumerate(self.stage3):
            if i==0:
                output3_0 = layer(output2_cat)
            elif i==1:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3)

        output3_cat = self.bn3(torch.cat([output3_0, output3], 1))
        output = self.classifier(output3_cat)
        return output