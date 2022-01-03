import pickle
from pathlib import PurePath
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import ANNOTATIONS, BOUNDING_BOXES, CLASS, HEIGHT, HULL, WIDTH


class Card:
    def __init__(self, fpath: PurePath) -> None:
        self._fpath = fpath
        self._image = self._load_image()
        self._value = None
        self._bounding_boxes = None
        self._hulls = None
        self._size = None
        self._load_annotations()

    def get_size(self) -> Tuple[int, int]:
        """Returns the size of the card in pixels (width, height)"""
        return self._size

    def get_value(self) -> str:
        return self._value

    def get_image(self):
        return self._image

    def numpy_image(self):
        return np.array(self._image)

    def get_radius(self):
        width, height = self._size
        return int(np.sqrt(width ** 2 + height ** 2) / 2)

    def get_fname(self):
        return self._fpath.stem

    def display(self):
        self._image.show()

    @property
    def value(self):
        return self._value

    def _load_image(self) -> Image.Image:
        return Image.open(self._fpath)

    def _load_annotations(self):
        annotations = pickle.load(
            open(
                self._fpath.parents[1].joinpath(
                    ANNOTATIONS + f"{self._fpath.stem}.pkl"
                ),
                "rb",
            )
        )
        self._value = annotations[CLASS]
        self._size = [annotations[WIDTH], annotations[HEIGHT]]
        self._bounding_boxes = annotations[BOUNDING_BOXES]
        self._hulls = annotations[HULL]

    def __repr__(self) -> str:
        return f"Card: {self._value} {self._size}"

    def __str__(self) -> str:
        return f"Card: {self._value} {self._size}"
