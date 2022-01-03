from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import PurePath
import pickle

from config import ANNOTATIONS, BOUNDING_BOXES, CLASS, HEIGHT, WIDTH, HULL


class Card:
    def __init__(self, fpath: PurePath) -> None:
        self.fpath = fpath
        self.image: np.array = self._load_image()
        self.value = None
        self.bounding_boxes = None
        self.hulls = None
        self.size = None
        self._load_annotations()

    def get_size(self) -> Tuple[int, int]:
        """Returns the size of the card in pixels (width, height)"""
        return self.size

    def get_radius(self):
        width, height = self.size
        return int(np.sqrt(width ** 2 + height ** 2) / 2)

    def get_fname(self):
        return self.fpath.stem

    def display(self):
        plt.axis("off")
        plt.imshow(self.image)
        plt.show()

    def _load_image(self) -> np.array:
        image = cv2.imread(str(self.fpath), cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    def _load_annotations(self):
        annotations = pickle.load(
            open(
                self.fpath.parents[1].joinpath(ANNOTATIONS + f"{self.fpath.stem}.pkl"),
                "rb",
            )
        )
        self.value = annotations[CLASS]
        self.size = [annotations[WIDTH], annotations[HEIGHT]]
        self.bounding_boxes = annotations[BOUNDING_BOXES]
        self.hulls =  annotations[HULL]

    def __repr__(self) -> str:
        return f"Card: {self.value} {self.size}"

    def __str__(self) -> str:
        return f"Card: {self.value} {self.size}"