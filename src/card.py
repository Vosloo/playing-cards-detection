# import mpimg
import json
from math import sqrt
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from config import ANNOTATIONS, BOUNDING_BOXES, HEIGHT, HULLS, WIDTH


class Card:
    def __init__(self, fname) -> None:
        self.fname = fname
        self.value: str = self._load_value()
        self.image: np.array = self._load_image()
        self.bounding_boxes, self.hulls, self.size = self._load_annotations()

    def get_size(self) -> Tuple[int, int]:
        """Returns the size of the card in pixels (width, height)"""
        return self.size

    def get_radius(self):
        width, height = self.size
        return int(sqrt(width ** 2 + height ** 2) / 2)

    def get_fname(self):
        return self.fname

    def display(self):
        plt.axis("off")
        plt.imshow(self.image)
        plt.show()

    def _load_value(self):
        return Path(self.fname).stem

    def _load_image(self) -> np.array:
        # return mpimg.imread(self.fname)
        image = cv2.imread(self.fname, cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    def _load_annotations(self):
        # Get path to directory containing annotations
        annotations = Path(self.fname).parents[1] / ANNOTATIONS
        with open(annotations / (Path(self.fname).stem + ".json")) as f:
            data = json.load(f)

        return data[BOUNDING_BOXES], data[HULLS], (data[WIDTH], data[HEIGHT])

    def __repr__(self) -> str:
        return f"Card: {self.value} {self.size}"

    def __str__(self) -> str:
        return f"Card: {self.value} {self.size}"
