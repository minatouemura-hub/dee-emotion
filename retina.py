import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Retina:
    """
        Args:
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l: a 2D Tensor of shape (B, 2). Contains normalized
            coordinates in the range [-1, 1].
        g: size of the first square patch.
        k: number of patches to extract in the glimpse.
        s: scaling factor that controls the size of
            successive patches.

    """
    def __init__(self,g,k,s):
        self.g = g
        self.k = k
        self.s = s
    def foveate(self,x,l):
        """
        Extract `k` square patches of size `g`, centered
        at location `l`. The initial patch is a square of
        size `g`, and each subsequent patch is a square
        whose side is `s` times the size of the previous
        patch.

        The `k` patches are finally resized to (g, g) and
        concatenated into a tensor of shape (B, k, g, g, C).
        """
        phi = []
        size = self.g
        # extract k patches of increasing size
        for i in range(self.k):
            patch = self.extract_patch(x,l,size)
            phi.append(patch)
            size = int(self.s * size)

        # resize the patches to squares of size g
        for i in range(1,len(phi)):
            k = phi[i].shape[-1]//self.g
            phi[i] = F.avg_pool2d(phi[i].float(),k)

         # concatenate into a single tensor and flatten
        phi = torch.cat(phi,1)#torch.Size([~~]*batch_size)
        phi = phi.view(phi.shape[0],-1)
        return phi#torch.Size([B,~])
    
    def extract_patch(self,x,l,size):
        #returns:patch a 4D Tensor of shape(B,size,size,C)
        B,C,H,W = x.shape
        start  =self.denormalize(H,l)#画像の比率として渡されるlをHで表す
        end = start + size
        
        #pad with zeros
        x = F.pad(x,(size // 2, size // 2 , size // 2 , size //2))#上下、左右に+size分だけパディング
        # loop through mini-batch and extract patches
        patch = []
        for i in range(B):
            patch.append(x[i,:,start[i,1]:end[i,1],start[i,0]:end[i,0]])
        return torch.stack(patch)

    def denormalize(self,T,coords):
        """
        Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, T] where `T` is
        the size of the image.
        """
        return (0.5 * ((coords + 1.0)*T)).long()
    def exeeds(self,from_x,to_x,from_y,to_y,T):
        if (from_x<0)or(from_y<0)or(to_x>T)or(to_y>T):
            return True
        else:
            False

class GlimpseNetwork(nn.Module):
    def __init__(self,h_g,h_l,g,k,s,c):
        super().__init__()

        self.retina = Retina(g,k,s)

        D_in = k*g*g*c
        self.fc1 = nn.Linear(D_in,h_g)

        D_in = 2
        self.fc2 = nn.Linear(D_in,h_l)

        self.fc3 = nn.Linear(h_g,h_g+h_l)
        self.fc4 = nn.Linear(h_l,h_g+h_l)
    
    def forward(self,x,l_t_prev):

        phi = self.retina.foveate(x,l_t_prev)

        #flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.shape[0],-1)

        phi_out = F.relu(self.fc1(phi))
        l_out = F.relu(self.fc2(l_t_prev))

        what = self.fc3(phi_out)
        where = self.fc4(l_out)

        g_t = F.relu(what + where)

        return g_t

class CoreNetwork(nn.Module):
    """
    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step. 

    Args:
        input_size: input size of the rnn.
        hidden_size: hidden size of the rnn.
        g_t: a 2D tensor of shape (B, hidden_size). The glimpse
            representation returned by the glimpse network for the
            current timestep `t`.
        h_t_prev: a 2D tensor of shape (B, hidden_size). The
            hidden state vector for the previous timestep `t-1`.

    Returns:
        h_t: a 2D tensor of shape (B, hidden_size). The hidden
            state vector for the current timestep `t`.
    """
    def __init__(self, input_size,hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size,hidden_size)
        self.h2h = nn.Linear(hidden_size,hidden_size)

    def forward(self,g_t,h_t_prev):
        h1 = self.i2h(g_t)
        h2 = self.h2h(h_t_prev)
        h_t = F.relu(h1+h2)
        return h_t

class ActionNetwork(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()

        self.fc = nn.Linear(input_size,output_size)

    def forward(self,h_t):
        a_t = F.log_softmax(self.fc(h_t),dim  =1)
        return a_t

class LocationNetwork(nn.Module):
    def __init__(self,input_size,output_size,std):
        super().__init__()

        self.std = std

        hid_size  = input_size // 2
        self.fc = nn.Linear(input_size,hid_size)
        self.fc_lt = nn.Linear(hid_size,output_size)
    
    def forward(self,h_t):
        feat  = F.relu(self.fc(h_t.detach()))
        mu = torch.tanh(self.fc_lt(feat))

        #reparametrizing trick
        l_t  =torch.distributions.Normal(mu,self.std).rsample()
        l_t = l_t.detach()
        log_pi = Normal(mu,self.std).log_prob(l_t)

        log_pi = torch.sum(log_pi, dim=1)
        l_t = torch.clamp(l_t,-1,1)

        return log_pi,l_t





    