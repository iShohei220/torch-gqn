import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from representation import Pyramid, Tower, Pool
from core import InferenceCore, GenerationCore
    
class GQN(nn.Module):
    def __init__(self, representation="pool", L=12, shared_core=False):
        super(GQN, self).__init__()
        
        # Number of generative layers
        self.L = L
                
        # Representation network
        self.representation = representation
        if representation=="pyramid":
            self.phi = Pyramid()
        elif representation=="tower":
            self.phi = Tower()
        elif representation=="pool":
            self.phi = Pool()
            
        # Generation network
        self.shared_core = shared_core
        if shared_core:
            self.inference_core = InferenceCore()
            self.generation_core = GenerationCore()
        else:
            self.inference_core = nn.ModuleList([InferenceCore() for _ in range(L)])
            self.generation_core = nn.ModuleList([GenerationCore() for _ in range(L)])
            
        self.eta_pi = nn.Conv2d(128, 2*3, kernel_size=5, stride=1, padding=2)
        self.eta_g = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0)
        self.eta_e = nn.Conv2d(128, 2*3, kernel_size=5, stride=1, padding=2)

    # EstimateELBO
    def forward(self, x, v, v_q, x_q, sigma):
        B, M, *_ = x.size()
        
        # Scene encoder
        if self.representation=='tower':
            r = x.new_zeros((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        for k in range(M):
            r_k = self.phi(x[:, k], v[:, k])
            r += r_k
            
        # Generator initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))

        # Inference initial state
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))
                
        elbo = 0
        for l in range(self.L):
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), 3, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)
            
            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x_q, v_q, r, c_e, h_e, h_g, u)
            
            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), 3, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)
            
            # Posterior sample
            z = q.rsample()
            
            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)
                
            # ELBO KL contribution update
            elbo -= torch.sum(kl_divergence(q, pi), dim=[1,2,3])
                
        # ELBO likelihood contribution update
        elbo += torch.sum(Normal(self.eta_g(u), sigma).log_prob(x_q), dim=[1,2,3])

        return elbo
    
    def generate(self, x, v, v_q):
        B, M, *_ = x.size()
        
        # Scene encoder
        if self.representation=='tower':
            r = x.new_zeros((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        for k in range(M):
            r_k = self.phi(x[:, k], v[:, k])
            r += r_k

        # Initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))
        
        for l in range(self.L):
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), 3, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)
            
            # Prior sample
            z = pi.sample()
            
            # State update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)
            
        # Image sample
        mu = self.eta_g(u)

        return torch.clamp(mu, 0, 1)
    
    def kl_divergence(self, x, v, v_q, x_q):
        B, M, *_ = x.size()

        # Scene encoder
        if self.representation=='tower':
            r = x.new_zeros((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        for k in range(M):
            r_k = self.phi(x[:, k], v[:, k])
            r += r_k
            
        # Generator initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))

        # Inference initial state
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))
                
        kl = 0
        for l in range(self.L):
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), 3, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)
            
            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x_q, v_q, r, c_e, h_e, h_g, u)
            
            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), 3, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)
            
            # Posterior sample
            z = q.rsample()
            
            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)
                
            # ELBO KL contribution update
            kl += torch.sum(kl_divergence(q, pi), dim=[1,2,3])

        return kl
    
    def reconstruct(self, x, v, v_q, x_q):
        B, M, *_ = x.size()

        # Scene encoder
        if self.representation=='tower':
            r = x.new_zeros((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        for k in range(M):
            r_k = self.phi(x[:, k], v[:, k])
            r += r_k
            
        # Generator initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))

        # Inference initial state
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))
                
        for l in range(self.L):
            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x_q, v_q, r, c_e, h_e, h_g, u)
            
            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), 3, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)
            
            # Posterior sample
            z = q.rsample()
            
            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)
                
        mu = self.eta_g(u)

        return torch.clamp(mu, 0, 1)
    