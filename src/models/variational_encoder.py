from typing import ForwardRef
import torch
from torch import optim
from torch.distributions import transforms
from torch.distributions.transforms import ReshapeTransform
from torchvision.datasets.mnist import MNIST; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import torchvision.datasets as datasets
from tqdm import tqdm 
import time
from torch.utils.data import DataLoader
import sys 
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

class VariationalEncoder_1(nn.Module):

    def __init__(self, latent_dims):
        super().__init__()
        #simple linear later 
        self.linear1_0 = nn.Linear(784, 512)
        self.linear2_0 = nn.Linear(512, 256)

        self.linear1_1 = nn.Linear(784, 512)
        self.linear2_1 = nn.Linear(512, 256)

        #calculate mean|
        #calculate varience
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)
        #adding the distribution to sample from 
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        x1 = x[0]
        x1 = torch.flatten(x1, start_dim=1)
        x1 = F.relu(self.linear1_0(x1))
        x1 = F.relu(self.linear2_0(x1))
        #second input layer
        x2 = x[1]
        x2 = torch.flatten(x2, start_dim=1)
        x2 = F.relu(self.linear1_1(x2))
        x2 = F.relu(self.linear2_1(x2))
        #calculate the mean 
        # print(x1.shape, x2.shape)
        x = torch.cat((x1,x2),dim= 1)
        # print(x.shape)
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

        self.encoder = VariationalEncoder_1(latentDims)
        self.decoder = Decoder(latentDims)


    def forward(self, x):

        z = self.encoder(x)
        return self.decoder(z)


if __name__=="__main__":
    sys.path.insert(1, "D:\Deep-Learning-2021\Deep-Learning-2021")
    from src.data.load_mnist import MNIST 
    latentDims = 10
    epochs = 10
    save_path = "D:\Deep-Learning-2021\Deep-Learning-2021\models\model_sum.pth"
    mnist_trainset = MNIST(root= "D:\Deep-Learning-2021\Deep-Learning-2021\src\data", train=True, transform = torchvision.transforms.ToTensor())
    vae = VariationalAutoEncoder(latentDims)
    opt = torch.optim.Adam(vae.parameters())
    train_loader = DataLoader(mnist_trainset, batch_size= 2 ) 
    sum_images = mnist_trainset.get_examples()
    # TODO : compatibility to GPU
    for epoch in tqdm(range(epochs)):
        for images , labels in tqdm(train_loader):
            opt.zero_grad()
            x_pred = vae(images)
            total = labels.sum()
            print(labels, total, type(total))
            loss = ((sum_images[total] - x_pred)**2).sum() + vae.encoder.kl
            loss.backward()
            opt.step()

    torch.save(vae.state_dict(), save_path)

