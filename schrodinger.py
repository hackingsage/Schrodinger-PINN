import torch
import torch.nn as nn
import scipy.constants as constants

class SchrodingerPINN(nn.Module):

    def __init__(self,layers,m):
        super(SchrodingerPINN,self).__init__()
        self.layers = nn.ModuleList()
        self.m = m
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i],layers[i+1]))
    
    def forward(self,x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        x = self.layers[-1](x)
        return x
    
    def compute_pde_residual(self,x,t):
        x.requires_grad_(True)
        t.requires_grad_(True)
        inputs = torch.cat((x,t),dim=1)
        psi = self.forward(inputs)
        dpsi_dt = torch.autograd.grad(psi,t,grad_outputs=torch.ones_like(psi),retain_graph=True,create_graph=True)[0]
        dpsi_dx = torch.autograd.grad(psi,x,grad_outputs=torch.ones_like(psi),retain_graph=True,create_graph=True)[0]
        d2psi_dx2 = torch.autograd.grad(dpsi_dx,x,grad_outputs=torch.ones_like(dpsi_dx),retain_graph=True,create_graph=True)[0]
        v = x ** 2 #Example Potential Function
        residue = (1j * constants.hbar * dpsi_dt) + ((constants.hbar**2)/(2*self.m)*d2psi_dx2) - (v * psi)
        return residue

    def loss_function(self,x,t):
        residue = self.compute_pde_residual(x,t)
        loss = torch.mean(torch.abs(residue)**2)
        return loss