# MNIST VAE Incorporating Perceptual Loss

This project demonstrates a VAE trained to generate MNIST-like, handwritten digits by incorporating [perceptual loss](https://arxiv.org/abs/1603.08155).  There are two models in this project: a VAE and a classification model used to compute perceptual loss in the VAE.  Both the VAE and the classification model are trained on MNIST and come from [Pytorch's example projects repo](https://github.com/pytorch/examples).

## Usage

`python main.py`

See `python main.py -h` for additional options.

I have included a trained classification model for convenience, so you don't need to run `classification.py` first.  The VAE will use the trained classification model for the perceptual loss.