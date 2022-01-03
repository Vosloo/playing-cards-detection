import pickle
import random

import matplotlib.image as mpimg

from background import Background
from config import IMAGES, BACKGROUNDS, BACKGROUNDS_PKL


class BackgroundFactory:
    def __init__(self) -> None:
        self.images = pickle.load(open(IMAGES + BACKGROUNDS + BACKGROUNDS_PKL, "rb"))
        self.no_images = len(self.images)

    def get_no_images(self):
        return self.no_images

    def get_random_background(self) -> Background:
        return Background(self.images[random.randint(0, self.no_images - 1)])

if __name__ == "__main__":
    bf = BackgroundFactory()
    
    avg_width = 0
    avg_height = 0
    for image in bf.images:
        bg = Background(image)
        height, width, _ = bg.get_size()
        avg_width += width
        avg_height += height

    avg_width /= bf.no_images
    avg_height /= bf.no_images

    print(avg_width, avg_height)