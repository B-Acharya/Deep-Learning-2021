# from _typeshed import OpenTextMode
from typing import ForwardRef
from matplotlib import image
import torch
from torch import optim
from torch.distributions import transforms
from torch.distributions.transforms import ReshapeTransform, StickBreakingTransform
from torch.nn.modules import activation
from torch.nn.modules.conv import ConvTranspose2d
from torch.nn.modules.pooling import MaxPool2d
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
from src.data.MNIST.mnist import sumNumber
from torch.utils.data import DataLoader
import sys 
import matplotlib.pyplot as plt
from torch.nn import BCELoss
from src.models.train_model import validate

# 
#from https://github.com/dstallmann/cell_cultivation_analysis/blob/1ef9c0e11e05200d672323f604674c3bc9e5e016/vae/VAE.py#L39
class UnFlatten(nn.Module):
	def forward(self, data):
		"""
		Unflattens the network at the beginning of the decoder
		@param data: data to be unflattened
		@return: unflattened data
		"""
		return data.view(data.size(0), -1, 1, 1)


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

############################################################################
#                                                                          #
#                   Code for more dense encoder                            #
#                                                                          #
############################################################################

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

class VariationalAutoEncoder(nn.Module):

    def __init__(self, latentDims) :
        super().__init__()

        self.encoder = VariationalEncoder_1(latentDims)
        self.decoder = Decoder(latentDims)


    def forward(self, x):

        z = self.encoder(x)
        return self.decoder(z)

############################################################################
#                                                                          #
#                   Code for more conv auto encoder                        #
#                                                                          #
############################################################################

class VariationalEncoderConv(nn.Module):

    def __init__(self, latent_dims):
        super().__init__()

        #conv layes
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels=6, kernel_size=5, stride=1),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.LeakyReLU(),
            nn.Linear(120, 84),
            nn.LeakyReLU(),
        )

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels=6, kernel_size=5, stride=1),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.LeakyReLU(),
            nn.Linear(120, 84),
            nn.LeakyReLU(),
        )

        #calculate mean|
        #calculate varience
        self.linear2 = nn.Linear(168, latent_dims)
        self.linear3 = nn.Linear(168, latent_dims)
        #adding the distribution to sample from 
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        x1 = x[0]
        x1 = self.encoder_1(x1.unsqueeze(0)) 
        #second brach 
        x2 = x[1]
        x2 = self.encoder_2(x2.unsqueeze(0)) 
        #second input layer
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
        # print(z.shape)
        return z

        
class DecoderConv(nn.Module) :

    def __init__(self, latentDims):
        super().__init__()
        #simple decoder layer
        self.linear1 = nn.Linear(latentDims, 128)
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 8, kernel_size=2, stride=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(968, 784),
            nn.Sigmoid(),
        ) 

    def forward(self, z):
        z = F.relu(self.linear1(z))
        # print(z.shape)
        #TODO why sigmoid ? 
        z = self.decoder(z)
        return z.reshape(-1,1,28,28)


class VariationalAutoEncoderConv(nn.Module):

    def __init__(self, latentDims) :
        super().__init__()

        self.encoder = VariationalEncoderConv(latentDims)
        self.decoder = DecoderConv(latentDims)

    def forward(self, x):

        z = self.encoder(x)
        return self.decoder(z)

#########################################################
#          conv with just one-encoder and decoder       #
#########################################################

class encoderConv(nn.Module):

    def __init__(self, latent_dims, image_channels=1, init_channels=8, device='cpu'):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels= image_channels, out_channels=init_channels, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Dropout(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=init_channels, out_channels=init_channels*2, kernel_size=4, stride=2, padding=1 ),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=init_channels*4, out_channels=init_channels*8, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(init_channels*8, 64)
        )
        #calculate mean
        self.linear2 = nn.Linear(64, latent_dims)
        #calculate varience
        self.linear3 = nn.Linear(64, latent_dims)
        #adding the distribution to sample from 
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        mu =  self.linear2(x)
        # calculate the varience TODO:why the exponent ?
        sigma = torch.exp(self.linear3(x))
        # the latent space to sample from
        z = mu + sigma*self.N.sample(mu.shape)
        # kl divergance for the latent space distribution and guassian normal distribution with mean = 0 and varience = 1 
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        # print(z.shape)
        return z

        
class decoderConv(nn.Module):

    def __init__(self, latentDims, init_channels=8, image_channels= 1):
        super().__init__()
        # simple decoder layer
        self.linear1 = nn.Linear(latentDims, 64)
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(64, init_channels*16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(init_channels*16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(init_channels*16, init_channels*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_channels*8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(init_channels*8, init_channels*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_channels*4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(init_channels*4, image_channels, kernel_size=4, stride=2, padding=3),
            )
    def forward(self, z):
        z = F.relu(self.linear1(z))
        # print(z.shape)
        return torch.sigmoid(self.decoder(z))


class VariationalAutoEncoderConv(nn.Module):

    def __init__(self, latentDims, device):
        super().__init__()
        self.encoder = encoderConv(latentDims, init_channels=16, image_channels=2, device=device)
        self.decoder = decoderConv(latentDims, init_channels=16, image_channels=2)

    def forward(self, x):

        z = self.encoder(x)
        return self.decoder(z)
#############################################################
#                   train the model                         #
#############################################################

def train(model, epochs, optimizer, train_dataloader, val_loader, device,output_iamges, save_path):
    train_loss = []
    val_loss = []
    prev_val_loss = np.inf
    lossFunc = BCELoss(reduction='sum')
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        counter = 0
        for images , labels in tqdm(train_dataloader):
            images = images.to(device)
            x_pred = model(images)
            # n1, n2 = get_digitis(labels)
            # sum_image = torch.cat([output_iamges[n1].unsqueeze(0), output_images[n2].unsqueeze(0)])
            # sum_image = sum_image.unsqueeze(1).to(device)
            # loss = ((output[total] - x_pred)**2).sum() + model.encoder.kl
            # loss = lossFunc(x_pred,images) + model.encoder.kl
            # print(labels)
            # fig , axs = plt.subplots(2,2)
            # axs[0, 0].imshow(images[0].squeeze(0)[0].cpu().numpy())
            # axs[0, 1].imshow(images[0].squeeze(0)[1].cpu().numpy())
            # axs[1, 0].imshow(output_images[labels][0].squeeze(0).cpu().numpy())
            # axs[1, 1].imshow(output_images[labels][1].squeeze(0).cpu().numpy())
            # plt.show()
            loss = lossFunc(x_pred, output_iamges[labels].unsqueeze(0).to(device)) + model.encoder.kl
            loss.backward()
            if counter%64 == 0 :
                optimizer.step()
                optimizer.zero_grad()
            counter += 1
            running_loss += loss.item()
        total_loss = running_loss/counter
        train_loss.append(total_loss)
        val = validate(model, val_loader, device, output_images, lossFunc)
        val_loss.append(val)
        print(f"train_loss -> epoch{epoch}: {total_loss:>7f} ")
        print(f"val_loss: {val:>7f} ")
        if val < prev_val_loss:
            torch.save(model.state_dict(), save_path + f"/model{epoch}.pth")
            prev_val_loss = val
    return train_loss[:], val_loss[:]

def get_examples(mnist):
    example = []
    for i in range(19):
        n1, n2 = get_digitis(i)
        image1 = mnist.train_data[mnist.train_labels== n1][0]/255
        image2 = mnist.train_data[mnist.train_labels== n2][0]/255
        example.append(torch.cat([image1.unsqueeze(0), image2.unsqueeze(0)]))
    return example

def get_digitis(number):
    if number//10 == 0:
        return 0, number
    else:
        return number//10, number%10


def save_loss_plot(train_loss, val_loss, learning_rate, latent_dims, epochs):
    plt.figure(figsize=(10,7))
    plt.plot(train_loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(f"/homes/bacharya/PycharmProjects/Deep-Learning-2021/reports/figures/vae_loss_{latent_dims}_{learning_rate}_{epochs}.jpg")


if __name__=="__main__":
    # sys.path.insert(1, "D:\Deep-Learning-2021\Deep-Learning-2021")
    from src.models.train_model import validate
    # torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    latentDims = 10
    epochs = 30  
    learning_rate = 0.001
    save_path = "/homes/bacharya/PycharmProjects/Deep-Learning-2021/models/"
    mnist_trainset = MNIST(root= "/homes/bacharya/PycharmProjects/Deep-Learning-2021/src/data/MNIST", train=True, download=True, transform = torchvision.transforms.ToTensor())
    mnist_val =  MNIST(root= "/homes/bacharya/PycharmProjects/Deep-Learning-2021/src/data/MNIST", train=False, download=True, transform = torchvision.transforms.ToTensor())
    train_loader = DataLoader(mnist_trainset, batch_size=2, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=2, shuffle=False)
    train_data = []
    train_label = []
    for data, label in train_loader:
        data = torch.cat([data[0], data[1]])
        label = torch.sum(label)
        train_data.append(data.unsqueeze(0))
        train_label.append(label.unsqueeze(0))
    train_data = torch.cat(train_data)
    train_label = torch.cat(train_label)
    train_dataset = sumNumber(train_data, train_label)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    ################################################
    val_data = []
    val_label = []
    for data, label in val_loader:
        data = torch.cat([data[0], data[1]])
        label = torch.sum(label)
        val_data.append(data.unsqueeze(0))
        val_label.append(label.unsqueeze(0))
    val_data = torch.cat(val_data)
    val_label = torch.cat(val_label)
    val_dataset = sumNumber(val_data, val_label)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    output_images = get_examples(mnist_trainset)
    print(output_images[0].shape)
    # plt.imshow(output_images[1][1])
    # plt.show()
    # print(len(next(iter(train_loader))))
    # print(vae)
    # output = vae(next(iter(train_loader))[0])
    vae = VariationalAutoEncoderConv(latentDims, device)
    vae = vae.to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    train_loss, val_loss = train(vae, epochs, opt, train_loader, val_loader, device, output_images, save_path=save_path)
    save_loss_plot(train_loss, val_loss, learning_rate, latentDims, epochs)
    # np.save(save_path + "\losses.npy", np.array(losses))
    # print(mnist_trainset[0])