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
parser.add_argument('--lr', type=int, default=.001, metavar='N',
                    help='learning_rate')
parser.add_argument('--batch-size', type=int, default=5, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
classify_model.eval()

model = CelebVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # Perceptual Distance

    x_activations = classify_model(x)
    recon_x_activations = classify_model(recon_x)

    PD = 0.0
    for (x_act, recon_x_act) in zip(x_activations, recon_x_activations):
        PD += .3 * F.mse_loss(x_act, recon_x_act, reduction='sum')

    MSE = .3* F.mse_loss(x, recon_x, reduction='sum')

    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # print(f'BCE: {BCE}, PD: {PD}, KLD: {KLD}')
    # return BCE + PD - KLD
    return PD + KLD + MSE
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


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    kwargs = {'num_workers': 3, 'pin_memory': False} if args.cuda else {}
    # train_loader, testloader = get_train_test_dataloaders(args.batch_size, kwargs)
    train_loader = get_training_dataloader(args.batch_size, kwargs, limit=1000, runtime=True)

    for epoch in range(1, args.epochs + 1):
        train(train_loader, epoch)

    torch.save(model, os.path.join(get_model_dir(), 'test5.pt'))
