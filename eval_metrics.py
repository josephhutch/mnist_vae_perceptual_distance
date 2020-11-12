import torch
import numpy as np
import torch.nn.functional as F
from visualization import load_model
from classifiers import VGG16PerceptualMeasurer
from skimage.measure import compare_ssim
from datasets import get_training_dataloader
from torchvision.models.inception import inception_v3


def perceptual_distance(images, r_images, device):
    classify_model = VGG16PerceptualMeasurer().to(device)
    classify_model = classify_model.eval()

    x_activations = classify_model(images)
    recon_x_activations = classify_model(r_images)

    PD = 0.0
    weights = [.161, 1, 1]
    for (x_act, recon_x_act) in zip(x_activations, recon_x_activations):
        for layer, r_layer, weight in zip(x_act, recon_x_act, weights):
            PD += weight * F.mse_loss(x_act, recon_x_act, reduction='sum')

    return PD


def ssim_loss(images, r_images):
    images = images.detach().numpy()
    r_images = r_images.detach().numpy()
    score = 0

    for image, r_image in zip(images, r_images):
        score += compare_ssim(image.reshape([224,224,3]), r_image.reshape([224,224,3]), multichannel=True, data_range=1)

    return score


def inception_loss(r_images):
    eps = 1*10**-16
    i_model = inception_v3(pretrained=True).eval()
    preds = i_model(r_images).detach().numpy()
    py = np.expand_dims(preds.mean(axis=0), 0)
    kl_d = preds * (np.log(preds + eps) - np.log(py + eps))
    sum_kl_d = kl_d.sum(axis=1)
    avg_kl_d = np.mean(sum_kl_d)
    is_score = np.exp(avg_kl_d)
    return is_score


if __name__ == "__main__":
    device = torch.device("cpu")
    dataloader = get_training_dataloader(10, {'num_workers':1}, limit=100, runtime=True)
    model = load_model('FullTestLong.pt', device).eval()
    batch_size = 20
    PD = 0
    SS = 0
    IL = 0
    for batch in dataloader:
        batch = batch.to(device)
        r_batch, _, _ = model(batch)
        PD += perceptual_distance(batch, r_batch, device)/batch_size
        SS += ssim_loss(batch, r_batch)/batch_size
        # IL += inception_loss(r_batch)/batch_size

    print('PD:', PD/10, "SS:", SS/10, "IL:", IL/10)

