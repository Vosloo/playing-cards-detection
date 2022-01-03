import os
import random
from typing import List

from config import IMAGES, LABELED, SCANS, TEST
from card import Card


class CardFactory:
    """
    A factory for creating card objects.
    """

    def __init__(self, dataset_path) -> None:
        self.images = self._load_images_names(dataset_path)
        self.no_images = len(self.images)

    def get_no_images(self):
        return self.no_images

    def get_random_cards(self, k=1) -> List[Card]:
        return [Card(fname) for fname in random.sample(self.images, k)]

    def _load_images_names(self, dataset_path):
        images = os.listdir(IMAGES + SCANS + dataset_path + LABELED)
        images = [IMAGES + SCANS + dataset_path + LABELED + item for item in images]
        return images

if __name__ == "__main__":
    cf = CardFactory(TEST)
    card, = cf.get_random_cards()
    print(card)
    print(card.get_fname(), card)
    print(card.get_size(), card.get_radius())
    card.display()