from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Permutation(ABC, nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
    
    @abstractmethod
    def forward(self, x, inverse=False):
        pass
    

class PermutationIdentity(Permutation):
    def forward(self, x, dim=1, inverse=False):
        return x
    

class PermutationFlip(Permutation):
    def __init__(self, seq_len):
        super().__init__(seq_len)       
        
    def forward(self, x, dim=1, inverse=False):
        # return torch.flip(x, dims=[dim])
        return x.flip(dims=[dim])
    

class PermutationShuffle(Permutation):
    def __init__(self, seq_len):
        super().__init__(seq_len)       
        
        shuffle_idx = torch.randperm(seq_len)
        reverse_idx = torch.argsort(shuffle_idx)
        
        self.register_buffer('shuffle_idx', shuffle_idx)
        self.register_buffer('reverse_idx', reverse_idx)
        
    def forward(self, x, dim=1, inverse=False):
        idx = self.reverse_idx if inverse else self.shuffle_idx
        
        return torch.index_select(x, dim=dim, index=idx)


PERMUTATION_TYPES = {
    "identity": PermutationIdentity,
    "flip": PermutationFlip,
    "shuffle": PermutationShuffle,
}


if __name__ == '__main__':
    B, C, H, W, P = 1, 3, 4, 4, 2
    seq_len = (H//P) * (W//P)
    patch_dim = C * P * P
    x = torch.rand([B, seq_len, patch_dim])
    
    # permutations = [
    #     PermutationIdentity(seq_len, dim=1),
    #     PermutationFlip(seq_len, dim=1),
    #     PermutationShuffle(seq_len, dim=1),
    # ]
    
    permutation_classes = [v for (k, v) in PERMUTATION_TYPES.items()]
    permutations = [perm_class(seq_len, dim=1) for perm_class in permutation_classes]
    
    for perm in permutations:
        print(f"\n--- Testing {perm.__class__.__name__} ---")
        x_copy = torch.randn_like(x)
        print(f"{x_copy}")
        x_perm = perm(x_copy)
        print(x_perm)
        x_recovered = perm(x_perm, inverse=True)
        print(x_recovered)
        
        is_identical = torch.allclose(x_copy, x_recovered, atol=1e-6)
        if is_identical:
            print("Success")
        else:
            print("Failure")
            