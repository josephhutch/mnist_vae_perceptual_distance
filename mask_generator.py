import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class ImageMCARGenerator:
    """
    Samples mask from component-wise independent Bernoulli distribution
    with probability of _pixel_ to be unobserved p.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, batch):
        gen_shape = list(batch.shape)
        num_channels = gen_shape[1]
        gen_shape[1] = 1
        bernoulli_mask_numpy = np.random.choice(2, size=gen_shape,
                                                p=[1 - self.p, self.p])
        bernoulli_mask = torch.from_numpy(bernoulli_mask_numpy).float()
        repeat_times = [1, num_channels] + [1] * (len(gen_shape) - 2)
        mask = bernoulli_mask.repeat(*repeat_times)
        return mask

class RectangleGenerator:
    """
    Generates for each object a mask where unobserved region is
    a rectangle which square divided by the image square is in
    interval [min_rect_rel_square, max_rect_rel_square].
    """
    def __init__(self, min_rect_rel_square=0.3, max_rect_rel_square=1):
        self.min_rect_rel_square = min_rect_rel_square
        self.max_rect_rel_square = max_rect_rel_square

    def gen_coordinates(self, width, height):
        x1, x2 = np.random.randint(0, width, 2)
        y1, y2 = np.random.randint(0, height, 2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        return int(x1), int(y1), int(x2), int(y2)

    def __call__(self, batch):
        batch_size, num_channels, width, height = batch.shape
        mask = torch.zeros_like(batch)
        for i in range(batch_size):
            x1, y1, x2, y2 = self.gen_coordinates(width, height)
            sqr = width * height
            while not (self.min_rect_rel_square * sqr <=
                       (x2 - x1 + 1) * (y2 - y1 + 1) <=
                       self.max_rect_rel_square * sqr):
                x1, y1, x2, y2 = self.gen_coordinates(width, height)
            mask[i, :, x1: x2 + 1, y1: y2 + 1] = 1
        return mask

class MixtureMaskGenerator:
    """
    For each object firstly sample a generator according to their weights,
    and then sample a mask from the sampled generator.
    """
    def __init__(self, generators, weights):
        self.generators = generators
        self.weights = weights

    def __call__(self, batch):
        w = np.array(self.weights, dtype='float')
        w /= w.sum()
        c_ids = np.random.choice(w.size, batch.shape[0], True, w)
        mask = torch.zeros_like(batch, device='cpu')
        for i, gen in enumerate(self.generators):
            ids = np.where(c_ids == i)[0]
            if len(ids) == 0:
                continue
            samples = gen(batch[ids])
            mask[ids] = samples
        return mask

class ImageMaskGenerator:

    def __init__(self):
        mcar = ImageMCARGenerator(0.95)
        common = RectangleGenerator()
        self.generator = MixtureMaskGenerator([mcar, common], [2, 2])