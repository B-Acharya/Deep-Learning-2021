from typing import ForwardRef
import torch
from torch import optim
from torch.distributions import transforms
from torch.distributions.transforms import ReshapeTransform; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import torchvision.datasets as datasets
from tqdm import tqdm 
import time
# 


# code adapted form https://avandekleut.github.io/vae/
class Encoder(nn.Module):

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



class Decoder(nn.Module) :

    def __init__(self, latentDims):
        super().__init__()
        #simple decoder layer

        self.linear1 = nn.Linear(latentDims, 512)
        self.linear2 = nn.Linear(512, 784)
        

    def forward(self, z):

        z = F.relu(self.linear1(z))
        #TODO why sigmoid ? 
        z = F.sigmoid(self.linear2(z))
        return z.reshape(-1,1,28,28)
        

class VariationalAutoEncoder(nn.Module):

    def __init__(self, latentDims) :
        super().__init__()

        self.encoder = VariationalEncoder(latentDims)
        self.decoder = Decoder(latentDims)


    def forward(self, x):

        z = self.encoder(x)
        return self.decoder(z)


if __name__=="__main__":

    latentDims = 2 
    epochs = 1
    save_path = "D:\Deep-Learning-2021\Deep-Learning-2021\models\model.pth"
    mnist_trainset = datasets.MNIST(root= "D:\Deep-Learning-2021\Deep-Learning-2021\src\data", train=True, download = True, transform = torchvision.transforms.ToTensor())
    vae = VariationalAutoEncoder(latentDims)
    opt = torch.optim.Adam(vae.parameters())
    #TODO : compatibility to GPU
    for epoch in tqdm(range(epochs)):
        for x , y in tqdm(mnist_trainset):
            opt.zero_grad()
            x_pred = vae(x)
            loss = ((x - x_pred)**2).sum() + vae.encoder.kl
            loss.backward()
            opt.step()
            # print(epoch)

    torch.save(vae.state_dict(), save_path)
    