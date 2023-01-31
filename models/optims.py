
import torch.optim as optim

from tools.library import OptimRegistry


@OptimRegistry.register('Adam')
class Adam(optim.Adam):
    pass 

    
