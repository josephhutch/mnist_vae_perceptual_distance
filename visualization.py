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
        sample = torch.randn(5, 20).to(device)
        sample = model.decode(sample).cpu()
        if not plot:
            return sample
        for s in sample:
            t = np.linspace(0,4,4*40)
            x = s.reshape([6,160]).detach().numpy()
            fig, ax = plt.subplots(nrows=1, ncols=1)
            for channel in x:
                ax.plot(t,channel)
            ax.set(xlabel='Time(s)', ylabel="Magnitude", title='Sampled Signals')
            plt.show()



def latentVisualization(filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(filename, device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    dataloader = get_training_dataloader(batch_size=1, kwargs=kwargs)

    z_x = []
    z_y = []
    z_z = []
    color = []
    colors = ['r','b','g','y']

    for data, label in dataloader:
        data = data.to(device)
        mu, logvar = model.encode(data)
        z = model.reparameterize(mu, logvar).tolist()[0]
        z_x.append(z[0])
        z_y.append(z[1])
        z_z.append(z[2])
        color.append(colors[int(label.item())])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z_x, z_y, z_z, c=color)
    plt.show()


def sample_and_closest_real(filename):
    samp = sample(filename, plot=False)
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    dataloader = get_training_dataloader(batch_size=1, kwargs=kwargs)
    dataset = dataloader.dataset.x
    lenght = len(dataset)
    t = np.linspace(0, 4, 4 * 40)
    for s in samp:
        rmse = torch.sqrt(torch.mean((s.repeat(lenght,1) - dataset)**2, axis=0))
        index = np.argmin(rmse)
        closest = dataset[index]
        s = s.reshape([6,160])
        closest = closest.reshape([6,160])
        fig, ax = plt.subplots(nrows=2, ncols=1)
        for channel1, channel2 in zip(s, closest):
            ax[0].plot(t, channel1)
            ax[1].plot(t, channel2)
        ax[1].set(xlabel='Time(s)', ylabel="Magnitude", title='Closest Real Signal')
        ax[0].set(xlabel='Time(s)', ylabel="Magnitude", title='Sampled Signal')
        plt.show()



if __name__ == '__main__':
    in_out_comparison('test5.pt')
    # latentVisualization('latent_test.pt')
    # sample('300_epoch_basic.pt')
    # sample_and_closest_real('300_epoch_basic.pt')