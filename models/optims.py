
import torch.optim as optim

from library import OptimRegistry


@OptimRegistry.register('Adam')
class Adam(optim.Adam):
    pass 

    
