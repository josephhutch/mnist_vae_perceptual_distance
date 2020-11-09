from datasets import get_training_dataloader
from mask_generator import *
import matplotlib.pyplot as plt
from visualization import load_model
import torch


def mask_tests():
    dataloader = get_training_dataloader(1, {'num_workers':1}, limit=50)
    mask_generator = ImageMaskGenerator()

    for image in dataloader:
        mask = mask_generator.generator(image)
        masked_img = image*mask
        masked_img = masked_img.detach().numpy().reshape([224,224,3])
        plt.imshow(masked_img)
        plt.show()


def test_model_with_masked_data():
    dataloader = get_training_dataloader(1, {'num_workers':1}, limit=50)
    mask_generator = ImageMaskGenerator()
    model = load_model(filename='test5.pt', device=torch.device('cpu'))

    for image in dataloader:
        mask = mask_generator.generator(image)
        masked_img = image*mask
        recon, mu, logvar = model(masked_img)
        recon = recon.detach().numpy().reshape([224,224,3])
        plt.imshow(recon)
        plt.show()



if __name__ == '__main__':
    # mask_tests()
    test_model_with_masked_data()

