import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from pathing import *
from datasets import  get_training_dataloader
from VAE import CelebVAE

def load_model(filename, device):
    path = os.path.join(get_model_dir(), filename)
    model = torch.load(path, map_location=device)
    return model.eval()


def in_out_comparison(filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(filename, device).cpu()
    dataloader = get_training_dataloader(batch_size=1, kwargs={'num_workers': 1}, limit=50)

    for data in dataloader:
        r_batch, mu, logvar = model(data)
        fig, ax = plt.subplots(nrows=2, ncols=1)
        data = data.detach().numpy()
        r_batch = r_batch.detach().numpy()
        data = data.reshape([224,224,3])
        r_batch = r_batch.reshape([224, 224, 3])
        ax[0].imshow(data)
        ax[1].imshow(r_batch)
        plt.show()


def sample(filename, plot=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(filename, device)

    with torch.no_grad():
        sample = torch.randn(5, 512).to(device)
        sample = model.decode(sample).cpu()
        if not plot:
            return sample
        for s in sample:
            x = s.reshape([224,224,3]).detach().numpy()
            plt.imshow(x)
            plt.show()



if __name__ == '__main__':
    in_out_comparison('Test2.pt')
    # sample('FullTestLong.pt')