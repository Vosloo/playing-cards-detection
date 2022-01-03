import pickle
import random
from typing import List

import matplotlib.image as mpimg

from background import Background
from config import IMAGES, BACKGROUNDS, BACKGROUNDS_PKL


class BackgroundFactory:
    def __init__(self) -> None:
        self.images = pickle.load(open(IMAGES + BACKGROUNDS + BACKGROUNDS_PKL, "rb"))
        self.no_images = len(self.images)

    def get_no_images(self):
        return self.no_images

    def get_random_backgrounds(self, k=1) -> List[Background]:
        return [Background(path) for path in random.sample(self.images, k=k)]

if __name__ == "__main__":
    bf = BackgroundFactory()
    bgs = bf.get_random_backgrounds(k=2)

    for bg in bgs:
        bg.display()