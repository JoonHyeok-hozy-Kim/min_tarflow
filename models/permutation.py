from abc import ABC, abstractmethod


class PermutationBase(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def forward(self, x, dim=1, inverse=False):
        pass


class PermutationIdentity(PermutationBase):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, dim=1, inverse=False):
        return x
    

class PermutationFlip(PermutationBase):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, dim=1, inverse=False):
        return x.flip(dims=[dim])


class PermutationShuffle(PermutationBase):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, dim=1, inverse=False):
        raise NotImplementedError("Shuffle operation is not implemented")