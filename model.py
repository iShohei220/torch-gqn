import torch
from torch import nn
from torch.nn import functional as F

from pixyz.distributions import Normal
from pixyz.losses import NLL, KullbackLeibler

from conv_lstm import Conv2dLSTMCell
from representation import Representation


# cores
class GeneratorCore(nn.Module):
    def __init__(self, v_dim, r_dim, z_dim, h_dim, SCALE):
        super(GeneratorCore, self).__init__()
        self.core = Conv2dLSTMCell(v_dim + r_dim + z_dim, h_dim, kernel_size=5, stride=1, padding=2)
        self.upsample = nn.ConvTranspose2d(h_dim, h_dim, kernel_size=SCALE, stride=SCALE, padding=0)
        
    def forward(self, z, v, r, h_g, c_g, u):
        h_g, c_g =  self.core(torch.cat([z, v, r], dim=1), [h_g, c_g])
        u = self.upsample(h_g) + u
        return h_g, c_g, u


class InferenceCore(nn.Module):
    def __init__(self, x_dim, v_dim, r_dim, h_dim):
        super(InferenceCore, self).__init__()
        self.core = Conv2dLSTMCell(2*h_dim + x_dim + v_dim + r_dim, h_dim, kernel_size=5, stride=1, padding=2)
        
    def forward(self, x, v, r, h_g, h_e, c_e, u):
        h_e, c_e = self.core(torch.cat([h_g, u, x, v, r], dim=1), [h_e, c_e])
        return h_e, c_e

# distributions
class Generator(Normal):
    def __init__(self, x_dim, h_dim):
        super(Generator, self).__init__(cond_var=["u", "sigma"],var=["x_q"])
        self.eta_g = nn.Conv2d(h_dim, x_dim, kernel_size=1, stride=1, padding=0)
        
    def forward(self, u, sigma):
        mu = self.eta_g(u)
        return {"loc":mu, "scale":sigma}

class Prior(Normal):
    def __init__(self, z_dim, h_dim):
        super(Prior, self).__init__(cond_var=["h_g"],var=["z"])
        self.z_dim = z_dim
        self.eta_pi = nn.Conv2d(h_dim, 2*z_dim, kernel_size=5, stride=1, padding=2)

    def forward(self, h_g):
        mu, logvar = torch.split(self.eta_pi(h_g), self.z_dim, dim=1)
        std = F.softplus(logvar)
        return {"loc":mu ,"scale":std}
    
class Inference(Normal):
    def __init__(self, z_dim, h_dim):
        super(Inference, self).__init__(cond_var=["h_i"],var=["z"])
        self.z_dim = z_dim
        self.eta_e = nn.Conv2d(h_dim, 2*z_dim, kernel_size=5, stride=1, padding=2)
        
    def forward(self, h_i):
        mu, logvar = torch.split(self.eta_e(h_i), self.z_dim, dim=1)
        std = F.softplus(logvar)
        return {"loc":mu, "scale":std}
    
class GQN(nn.Module):
    def __init__(self, x_dim, v_dim, r_dim, h_dim, z_dim, L, SCALE):
        super(GQN, self).__init__()
        self.L = L
        self.h_dim = h_dim
        self.SCALE = SCALE

        self.phi = Representation(x_dim, v_dim, r_dim)
        self.generator_core = GeneratorCore(v_dim, r_dim, z_dim, h_dim, self.SCALE)
        self.inference_core = InferenceCore(x_dim, v_dim, r_dim, h_dim)

        self.upsample   = nn.ConvTranspose2d(h_dim, h_dim, kernel_size=SCALE, stride=SCALE, padding=0)
        self.downsample_x = nn.Conv2d(x_dim, x_dim, kernel_size=SCALE, stride=SCALE, padding=0)
        self.downsample_u = nn.Conv2d(h_dim, h_dim, kernel_size=SCALE, stride=SCALE, padding=0)

        # distribution
        self.pi = Prior(z_dim, h_dim)
        self.q = Inference(z_dim, h_dim)
        self.g = Generator(x_dim, h_dim)

    def forward(self, x, v, v_q, x_q, sigma):
        batch_size, n_views, _, h, w = x.size()
        
        # Merge batch and view dimensions.
        _, _, *x_dims = x.size()
        _, _, *v_dims = v.size()

        x = x.view((-1, *x_dims))
        v = v.view((-1, *v_dims))

        # representation generated from input images
        # and corresponding viewpoints
        r = self.phi(x, v)

        # Seperate batch and view dimensions
        _, *r_dims = r.size()
        r = r.view((batch_size, n_views, *r_dims))

        # sum over view representations
        r = torch.sum(r, dim=1)

        _, _, h, w = x.size()

        # Increase dimensions
        v_q = v_q.view(batch_size, -1, 1, 1).repeat(1, 1, h//self.SCALE, w//self.SCALE)
        
        if r.size(2) != h//self.SCALE:
            r = r.repeat(1, 1, h//self.SCALE, w//self.SCALE)

        # Reset hidden state
        hidden_g = x_q.new_zeros((batch_size, self.h_dim, h//self.SCALE, w//self.SCALE))
        hidden_i = x_q.new_zeros((batch_size, self.h_dim, h//self.SCALE, w//self.SCALE))

        # Reset cell state
        cell_g = x_q.new_zeros((batch_size, self.h_dim, h//self.SCALE, w//self.SCALE))
        cell_i = x_q.new_zeros((batch_size, self.h_dim, h//self.SCALE, w//self.SCALE))
        
        u = x.new_zeros((batch_size, self.h_dim, h, w))
        
        _x_q = self.downsample_x(x_q)

        kls = 0
        for _ in range(self.L):
            # kl
            z = self.q.sample({"h_i": hidden_i}, reparam=True)["z"]
            kl = KullbackLeibler(self.q, self.pi)
            kl_tensor = kl.estimate({"h_i":hidden_i, "h_g":hidden_g})
            kls += kl_tensor
            # update state
            _u = self.downsample_u(u)
            hidden_i, cell_i = self.inference_core(_x_q, v_q, r, hidden_g, hidden_i, cell_i, _u)
            hidden_g, cell_g, u = self.generator_core(z, v_q, r, hidden_g, cell_g, u)
            
        x_q_rec = torch.clamp(self.g.sample_mean({"u": u, "sigma":sigma}), 0, 1)
        nll = NLL(self.g)
        nll_tensor = nll.estimate({"u":u, "sigma":sigma, "x_q": x_q})

        return nll_tensor, kls, x_q_rec
    
    def generate(self, x, v, v_q):
        batch_size, n_views, _, h, w = x.size()
        
        # Merge batch and view dimensions.
        _, _, *x_dims = x.size()
        _, _, *v_dims = v.size()

        x = x.contiguous().view((-1, *x_dims))
        v = v.contiguous().view((-1, *v_dims))

        # representation generated from input images
        # and corresponding viewpoints
        r = self.phi(x, v)

        # Seperate batch and view dimensions
        _, *r_dims = r.size()
        r = r.view((batch_size, n_views, *r_dims))

        # sum over view representations
        r = torch.sum(r, dim=1)

        # Increase dimensions
        v_q = v_q.view(batch_size, -1, 1, 1).repeat(1, 1, h//self.SCALE, w//self.SCALE)
        
        if r.size(2) != h//self.SCALE:
            r = r.repeat(1, 1, h//self.SCALE, w//self.SCALE)

        # Reset hidden state
        hidden_g = x.new_zeros((batch_size, self.h_dim, h//self.SCALE, w//self.SCALE))

        # Reset cell state
        cell_g = x.new_zeros((batch_size, self.h_dim, h//self.SCALE, w//self.SCALE))
        
        u = x.new_zeros((batch_size, self.h_dim, h, w))
        
        for _ in range(self.L):
            # kl
            z = self.pi.sample({"h_g": hidden_g})["z"]
            # update state
            hidden_g, cell_g, u = self.generator_core(z, v_q, r, hidden_g, cell_g, u)
            
        x_q_hat = torch.clamp(self.g.sample_mean({"u": u, "sigma":sigma}), 0, 1)

        return x_q_hat