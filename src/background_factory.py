import pickle
import random

import matplotlib.image as mpimg

DATA_DIR = "data/"
BACKGROUNDS = "backgrounds/"
BACKGROUNDS_PKL = "backgrounds.pkl"


class BackgroundFactory:
    def __init__(self) -> None:
        self.images = pickle.load(open(DATA_DIR + BACKGROUNDS + BACKGROUNDS_PKL, "rb"))
        self.no_images = len(self.images)

    def get_no_images(self):
        return self.no_images

    def get_random_background(self):
        return mpimg.imread(self.images[random.randint(0, self.no_images - 1)])
