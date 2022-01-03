import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Background:
    def __init__(self, fname) -> None:
        self._fname = fname
        self._image = self._load_img()

    def display(self):
        self._image.show()

    def get_size(self):
        return self._image.size

    def get_image(self):
        return self._image

    def get_fname(self):
        return self._fname

    def resize(self, scene_size: tuple):
        max_size = max(scene_size)

        self._image = self._image.resize((max_size, max_size))

    def save_background(self, fname):
        self._image.save(fname)

    @property
    def dtype(self):
        return self._image.dtype

    @property
    def shape(self):
        return self.get_size()

    def _load_img(self) -> Image.Image:
        return Image.open(self._fname)

    def __getitem__(self, item):
        return self._image[item]
