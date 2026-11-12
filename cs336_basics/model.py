import torch
import torch.nn as nn
class Linear(nn.Module):
    def __init__(self,in_features, out_features,device=None,dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features,in_features,device=device,dtype=dtype)) #according to adapters.py weight shld be of this shape(check run_linear fn in adapters.py)
        sigma = (2/(in_features+out_features))**0.5
        torch.nn.init.trunc_normal_(self.weight,std=sigma,a=(-3)*sigma,b=3*sigma)
    def forward(self,x):
        out = torch.matmul(x,self.weight.transpose(-1,-2))
        return out

