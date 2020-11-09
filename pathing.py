import os


def get_proj_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_model_dir():
    return os.path.join(get_proj_dir(), 'models')


def get_data_dir():
    return os.path.join(get_proj_dir(), 'data')


def get_celeb_dir():
    return os.path.join(get_data_dir(), 'Img')


def get_aligned_celeb_dir():
    return os.path.join(get_celeb_dir(), 'img_align_celeba')