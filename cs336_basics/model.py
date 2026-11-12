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
    
class Embedding(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_matrix = nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.embedding_matrix,std=1,a=(-3),b=3)
    def forward(self,x):
        return self.embedding_matrix[x] #embedding_matrix-->[num_embeddings,embedding_dim],x-->[batch,seq]
        #embedding_matrix[x] chooses the num_embeddings and replaces that dimension with batch,seq
        #therefore embedding_matrix[x]--->[batch,seq,embedding_dim]

class RMSNorm(nn.Module):
    def __init__(self,d_model: int,eps: float = 1e-5,device = None, dtype= None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(size=(d_model,),device=device,dtype=dtype))#the paper says the intial value of RMSNorm is 1 (3.3.1) so the gain shld started with a tensor full of ones
    def forward(self,x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.sum(((x**2)/self.d_model),dim=-1,keepdim=True)+self.eps)
        rmsnorm = (x/rms)*self.gain
        return rmsnorm.to(in_dtype)
    


    

