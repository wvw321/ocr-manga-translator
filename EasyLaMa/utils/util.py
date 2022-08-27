import os
import sys
import requests
import torch
from time import time
from torchvision import transforms
from PIL import Image
from urllib.parse import urlparse
from torch.hub import download_url_to_file, get_dir

to_tensor = transforms.ToTensor()

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
)


def download_model(url=LAMA_MODEL_URL):
    '''from https://github.com/Sanster/lama-cleaner/blob/main/lama_cleaner/helper.py'''
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(os.path.join(model_dir, "hub", "checkpoints"))
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)
    return cached_file


@torch.inference_mode()
def image_to_tensor(image, device="cpu"):
    image = to_tensor(image).to(device).to(torch.float)
    return image


def load_image(image):
    if isinstance(image, (str, os.PathLike)):
        if os.path.isfile(image):
            image = Image.open(image)
        elif isinstance(image, str):
            try:
                with requests.get(image, stream=True) as req:
                    image = Image.open(req.raw)
            except Exception as e:
                print(f"Can not find {image}")
                raise e
    return image.convert("RGB")


def load_images(images):
    return [load_image(image) for image in images]


@torch.inference_mode()
def pad(tensor, mod, **kwargs):
    c, h, w = tensor.shape
    if h % mod == 0:
        oh = h
    else:
        oh = (h // mod + 1) * mod

    if w % mod == 0:
        ow = w
    else:
        ow = (w // mod + 1) * mod

    pad_tensor = transforms.Pad(padding=(0, 0, ow - w, oh - h), **kwargs)
    padded = pad_tensor(tensor)
    return padded


class Timer:
    def __enter__(self):
        self.start = time()

    def __exit__(self, *args, **kwargs):
        self.time = time() - self.start
        print(self.time)
