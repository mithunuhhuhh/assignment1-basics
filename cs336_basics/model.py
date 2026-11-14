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
    
def silu(x):
    return x*torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff,device=None,dtype=None):
        super().__init__()
        self.w1=Linear(d_model,d_ff,device=device,dtype=dtype) 
        self.w2=Linear(d_ff,d_model,device=device,dtype=dtype) #the weights shape --> (out,in) [opposite to the linear layer] -->(d_model,d_ff) this is perfect for swiglu as we need w2 to be of that shape
        self.w3=Linear(d_model,d_ff,device=device,dtype=dtype) 
    def forward(self,x):
        return self.w2(silu(self.w1(x))*self.w3(x))

class RoPE(nn.Module): #RotaryPositionalEmbedding
    def __init__(self,theta: float,d_k: int,max_seq_len: int,device=None):
        super().__init__()
        self.i = torch.arange(0,max_seq_len,device=device)
        self.k = torch.arange(1,(d_k//2)+1,device=device)
        self.speed = 1/theta**(((2*self.k)-2)/d_k)
        self.angle_grid = torch.outer(self.i,self.speed) #creates a combined grid
        self.register_buffer("cosine",torch.cos(self.angle_grid),persistent=False) #applies cosine to every element
        self.register_buffer("sine",torch.sin(self.angle_grid),persistent=False) #applies sine to every element
    def forward(self,x,token_positions): #x is of the shape (batch,heads,sequence,d_k) d_k is the size of  each query/key vector,heads as in the no, of heads in teh multiattention block
        cos = self.cosine[token_positions]
        sin = self.sine[token_positions]
        x_even = x[...,::2] # indices : 0,2,4
        x_odd = x[...,1::2] # indices : 1,3,5
        new_first = x_even*cos-x_odd*sin
        new_second = x_even*sin+x_odd*cos
        new_x = torch.empty_like(x)
        new_x[...,::2] = new_first
        new_x[...,1::2] = new_second #... --> Ellipsis can fit any dimension it is flexible. 
        return new_x

        


    

