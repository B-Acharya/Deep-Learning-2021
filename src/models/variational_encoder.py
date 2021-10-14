from typing import ForwardRef
import torch
from torch.distributions.transforms import ReshapeTransform; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
# 


# code adapted form https://avandekleut.github.io/vae/
class Encoder(nn.module):

    def __init__(self, latentDims):
        super().__init__()
        #simple linear layer 
        self.linear1 = nn.Linear(784,512)
        self.linear2 = nn.Linear(512,latentDims)
        
    def forward(self, x):

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class VariationalEncoder(nn.Module):

    def __init__(self, latent_dims):
        super().__init__()
        #simple linear later 
        self.linear1 = nn.Linear(784, 512)
        #calculate mean|
        self.linear2 = nn.Linear(512, latent_dims)
        #calculate varience
        self.linear3 = nn.Linear(512, latent_dims)

        #adding the distribution to sample from 
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        #calculate the mean 
        mu =  self.linear2(x)
        #calculate the varience TODO:why the exponent ? 
        sigma = torch.exp(self.linear3(x))
        #the latent space to sample from 
        z = mu + sigma*self.N.sample(mu.shape)
        # kl divergance for the latent space distribution and guassian normal distribution with mean = 0 and varience = 1 
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z



class Decoder(nn.module) :

    def __init__(self, latentDims):
        super().__init__()
        #simple decoder layer

        self.linear1 = nn.Linear(latentDims, 512)
        self.linear2 = nn.Linear(512, 784)
        

    def forward(self, z):

        z = F.relu(self.linear1(z))
        z = F.sigmoid(self.linear2(z))
        return z.reshape(-1,1,28,28)
        

class VariationalAutoEncoder(nn.module):

    def __init__(self, latentDims) :
        super().__init__()

        self.encoder = VariationalEncoder(latentDims)
        self.decoder = Decoder(latentDims)


    def forward(self, x):

        z = self.encoder(x)
        return self.decoder(z)


if __name__=="__main__":

    pass 
