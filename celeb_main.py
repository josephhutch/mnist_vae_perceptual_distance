from __future__ import print_function
import classification
import os
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from datasets import get_train_test_dataloaders, CelebAData, get_training_dataloader
from VAE import CelebVAE
from classifiers import VGG16PerceptualMeasurer
from pathing import *
import matplotlib.pyplot as plt
import pdb
# from torchviz import make_dot


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=12, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

classify_model = VGG16PerceptualMeasurer().to(device)
classify_model = classify_model.eval()

# model = CelebVAE().to(device)
model = torch.load(os.path.join(get_model_dir(), 'Test2.pt'))
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # Perceptual Distance

    x_activations = classify_model.use_convs(x)
    recon_x_activations = classify_model.use_convs(recon_x)

    PD = 0.0
    weights = [.01,.01,.01,.01]
    for (x_act, recon_x_act) in zip(x_activations, recon_x_activations):
        for layer, r_layer, weight in zip(x_act, recon_x_act, weights):
            PD += weight * F.mse_loss(x_act, recon_x_act, reduction='sum')

    PD /= len(weights)
    MSE = .3*F.mse_loss(x, recon_x, reduction='sum')

    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return PD + KLD + MSE
    # return MSE
    # return PD


def train(train_loader, epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


if __name__ == "__main__":
    kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}
    # train_loader, testloader = get_train_test_dataloaders(args.batch_size, kwargs)
    train_loader = get_training_dataloader(args.batch_size, kwargs, limit=10000, runtime=True)

    for epoch in range(1, args.epochs + 1):
        train(train_loader, epoch)

    torch.save(model, os.path.join(get_model_dir(), 'Test2.pt'))
