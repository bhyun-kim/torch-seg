import torch.nn as nn 
import torch.nn.functional as F 

from tools.library import HeadRegistry

@HeadRegistry.register('Interpolate')
class Interpolate(nn.Module):
    def __init__(self, 
                 loss,
                 in_channels,
                 num_classes,
                 kernel_size=1,
                 size=None,
                 scale_factor=None, 
                 mode='nearest', 
                 align_corners=None,
                 recompute_scale_factor=None,
                 antialias=False
    ):
        """
        Args:
            size 
            num_classes
            scale_factor 
            mode (str) 
            align_corners (bool) 
            recompute_scale_factor () 
            antialias (bool) 
            
        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, num_classes, (kernel_size, kernel_size), bias=False)

        self.args_interpolate = dict(
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias
        )
        self.criterion = loss

        self.initialize_layers()

    def forward(self, input, labels):
        output = self.predict(input)
        return self.criterion(output, labels)

    def predict(self, input):
        out = self.conv(input)
        return F.interpolate(input=out, **self.args_interpolate)


    def initialize_layers(self):
        
        # init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname in ['Conv2d', 'Linear']:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()




@HeadRegistry.register('PixelShuffle')
class PixelShuffle(nn.Module):
    def __init__(self, 
                 loss,
                 in_channels,
                 num_classes,
                 kernel_size=1,
                 pixelshuffle_factor=2,
                 size=None,
                 scale_factor=None, 
                 mode='nearest', 
                 align_corners=None,
                 recompute_scale_factor=None,
                 antialias=False
    ):
        """
        Args:
            size 
            num_classes
            scale_factor 
            mode (str) 
            align_corners (bool) 
            recompute_scale_factor () 
            antialias (bool) 
            
        """
        super().__init__()

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=pixelshuffle_factor)

        self.conv = nn.Conv2d(
            in_channels // (pixelshuffle_factor ** 2),
            num_classes, 
            (kernel_size, kernel_size), 
            bias=False)

        self.args_interpolate = dict(
            size=size,
            scale_factor=scale_factor // pixelshuffle_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias
        )
        self.criterion = loss

        self.initialize_layers()

    def forward(self, input, labels):
        output = self.predict(input)
        return self.criterion(output, labels)

    def predict(self, input):
        out = self.pixel_shuffle(input)
        out = self.conv(out)
        return F.interpolate(input=out, **self.args_interpolate)


    def initialize_layers(self):
        
        # init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname in ['Conv2d', 'Linear']:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()