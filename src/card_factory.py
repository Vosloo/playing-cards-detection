import random
from typing import List

from config import ROOT_PATH, IMAGES, LABELED, SCANS, TEST
from card import Card


class CardFactory:
    """
    A factory for creating card objects.
    """

    def __init__(self, dataset_path) -> None:
        self.images_paths = self._load_images_paths(dataset_path)
        self.no_images = len(self.images_paths)

    def get_no_images(self):
        return self.no_images

    def get_random_cards(self, k=1) -> List[Card]:
        return [Card(path) for path in random.sample(self.images_paths, k)]

    def _load_images_paths(self, dataset_path):
        paths = ROOT_PATH.joinpath(IMAGES + SCANS + dataset_path + LABELED).glob("**/*")
        images = [p for p in paths if p.is_file()]
        return images


if __name__ == "__main__":
    cf = CardFactory(TEST)
    cf.get_random_cards()

