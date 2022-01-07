import random
from typing import List

from config import ROOT_PATH, IMAGES, LABELED, SCANS, TEST
from card import Card


class CardFactory:
    """
    A factory for creating card objects.
    """

    def __init__(self, dataset_path=TEST) -> None:
        self.images_paths = self._load_images_paths(dataset_path)
        self.no_images = len(self.images_paths)
        self.images_to_sample = set(range(self.no_images))

    def get_no_images(self):
        return self.no_images

    def get_random_cards(self, k=1) -> List[Card]:
        if not self.images_to_sample or k > len(self.images_to_sample):
            self.images_to_sample = set(range(self.no_images))

        taken = random.sample(self.images_to_sample, k)
        self.images_to_sample -= set(taken)

        return [Card(self.images_paths[ind]) for ind in taken]

    def _load_images_paths(self, dataset_path):
        paths = ROOT_PATH.joinpath(IMAGES + SCANS + dataset_path + LABELED).glob("**/*")
        images = [p for p in paths if p.is_file()]
        return images


if __name__ == "__main__":
    cf = CardFactory(TEST)
    (card,) = cf.get_random_cards()
    card.display()
