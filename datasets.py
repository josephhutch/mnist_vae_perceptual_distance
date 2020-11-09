import os
import pickle
import numpy as np
from progressbar import progressbar
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from pathing import get_aligned_celeb_dir, get_celeb_dir
import matplotlib.pyplot as plt


class CelebAData(Dataset):

    def __init__(self, limit):
        dir = get_aligned_celeb_dir()
        self.data = None
        for imgfile in progressbar(os.listdir(dir)[0:limit]):
            img = np.array(Image.open(os.path.join(dir, imgfile)).resize([224,224]), dtype=np.float32)
            img = img.reshape([1, 3, 224, 224]) / 255
            if self.data is None:
                self.data = img
            else:
                self.data = np.concatenate([self.data, img], axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class RuntimeCelebAData(Dataset):

    def __init__(self, limit=2000, sample=False):
        dir = get_aligned_celeb_dir()
        files = os.listdir(dir)
        self.images = dict()
        if sample:
            idx = np.random.choice(20000, limit)
            self.files = [os.path.join(dir, file) for file in files[idx]]
        else:
            self.files = [os.path.join(dir, file) for file in files[:limit]]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        if item in self.images.keys():
            return self.images[item]
        else:
            imgfile = self.files[item]
            img = np.array(Image.open(imgfile).resize([224, 224]), dtype=np.float32)
            img = img.reshape([3, 224, 224]) / 255
            self.images[item] = img
            return img



def pickle_dataset(filename, limit):
    dataset = CelebAData(limit)
    with open(os.path.join(get_celeb_dir(), filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def get_pickled_dataset(filename):
    return pickle.load(open(os.path.join(get_celeb_dir(), filename),'rb'))


def get_training_dataloader(batch_size, kwargs, limit=1000, runtime=False):
    if runtime:
        dataset = RuntimeCelebAData(limit)
    else:
        dataset = CelebAData(limit)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)
    return dataloader


def get_train_test_dataloaders(batch_size, kwargs, filename=None, limit=1000):
    if filename is None:
        dataset = CelebAData(limit)
    else:
        dataset = get_pickled_dataset(filename)

    lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]
    trainDS, testDS = random_split(dataset, lengths)

    trainloader = DataLoader(
            trainDS, batch_size=batch_size, shuffle=True, **kwargs)
    testloader = DataLoader(
            testDS, batch_size=batch_size, shuffle=True, **kwargs)
    return (trainloader, testloader)




if __name__ == '__main__':
    # pickle_dataset('complete_celebA.pickle', 2000)
    # data = get_pickled_dataset('complete_celebA.pickle')
    # data
    dir = get_aligned_celeb_dir()
    for imgfile in os.listdir(dir):
        img = Image.open(os.path.join(dir, imgfile))
        img = img.resize([128,128])
        plt.imshow(img)
        plt.show()