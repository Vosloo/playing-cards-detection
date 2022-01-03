from typing import List

from matplotlib import patches, pyplot as plt

from background import Background
from card import Card


class Scene:
    def __init__(self, bg: Background, *cards: Card) -> None:
        self.bg = bg
        self.cards = [*cards]
        self.scene = None # TODO: Implement

    def display(self):
        plt.axis("off")
        plt.imshow(self.scene)
        plt.show()

    def generate_scene(self):
        raise NotImplementedError

    def get_final(self):
        return self.final
