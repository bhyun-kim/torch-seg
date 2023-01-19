import torch.nn as nn 

from library import FrameRegistry 

@FrameRegistry.register('ModelFramer')
class ModelFramer(nn.Module):

    def __init__(self,
                 encoder,
                 head,
                 decoder=None,
                 ):
        super().__init__()

        self.encoder = encoder 
        self.decoder = decoder 
        self.head = head 

    def forward(self, input, labels=None):
        
        feat = self.encoder(input)

        if self.decoder: 
            feat = self.decoder(feat)

        if labels != None:
            
            feat = self.head(feat, labels) 
        else: 
            feat = self.head.predict(feat) 
        
        return feat
