import torch.nn as nn

class HookPoint(nn.Module):
    '''
    From Neel et al. : "Progress Measures for Grokking via Mechanistic Interpretability"
    A helper class to get access to intermediate activations (inspired by Garcon)
    It's a dummy module that is the identity function by default
    I can wrap any intermediate activation in a HookPoint and get a convenient way to add PyTorch hooks
    '''
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []
        self.moments_hooks = []
        self.inject_hooks = []
    
    def give_name(self, name):
        # Called by the model at initialisation
        self.name = name
    
    def add_hook(self, hook, dir='fwd'):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output, 
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):            
            return hook(module_output, name=self.name)
        
        def moments_hook(module, module_input, module_output):
            if isinstance(module_input, torch.Tensor):
                data = module_input
            elif isinstance(module_input, tuple) and len(module_input) == 1:
                data = module_input[0]
            else:
                raise TypeError(f"Unseen module_input type : {type(module_input)}")
            
            var, mean = torch.var_mean(data, dim=-1, keepdim=True, unbiased=False)
            result = {'var': var, 'mean': mean}
            # return hook(result, name=self.name)
            hook(result, name=self.name)
            return module_output
        
        def neutralize_hook(module, module_input, module_output):
            if self.neutralize_target_vectors is None:
                return module_output
            
            x = module_output            
            for v in self.neutralize_target_vectors:
                v = v / (v.norm() + 1e-9) 
                # Proj_v(x) = (x · v) * v
                proj = torch.einsum('bsh, h -> bs', x, v).unsqueeze(-1) * v
                x = x - proj
            return x
        
        def inject_hook(module, module_input, module_ouput):
            return hook(module_ouput, name=self.name)
        
        if dir=='fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir=='moments':
            handle = self.register_forward_hook(moments_hook)
            self.moments_hooks.append(handle)
        elif dir == 'neutralize':
            handle = self.register_forward_hook(neutralize_hook)
            self.fwd_hooks.append(handle)
        elif dir == 'inject':
            handle = self.register_forward_hook(inject_hook)
            self.inject_hooks.append(handle)
        elif dir=='bwd':
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")
    
    def remove_hooks(self, dir='fwd'):
        if (dir=='fwd') or (dir=='both'):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir=='bwd') or (dir=='both'):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")
    
    def forward(self, x):
        return x
    
    
