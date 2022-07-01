import torch
import torch.nn as nn

class APNet(nn.Module):
    """
    Action Proposal Network with Linear layers for box stacking and rope-box tasks.
    
    Input: latent sample from a VAE of dim latent_dim.
    Output: pick_x, pick_y, place_x, place_y (for box), or action_type, pick_x, pick_y, place_x, place_y (for rope-box)
    """
    def __init__(self, opt, trained_params=None):
        super().__init__()
        self.opt = opt
        self.dims = opt['dims'] 
        self.device = opt['device']
        self.dropout = opt['dropout']
        
        self.net = nn.Sequential()
        for i in range(len(self.dims) - 1):        
            self.net.add_module('lin' + str(i), nn.Linear(
                    self.dims[i], self.dims[i+1]))
            if i != len(self.dims) - 2:
                self.net.add_module('relu' + str(i), nn.ReLU())
                
    def forward(self, latent1, latent2):
        input = torch.cat([latent1, latent2], dim=1)
        return self.net(input)
