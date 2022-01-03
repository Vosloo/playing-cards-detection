from typing import List

import cv2
import numpy as np
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt

from background import Background
from background_factory import BackgroundFactory

from card import Card
from card_factory import CardFactory

from config import TEST


class Scene:
    def __init__(self, bg: Background, cards: List[Card]) -> None:
        self.bg = bg
        self.cards = cards
        self.scene = None

        self.max_width = max([card.get_size()[0] for card in self.cards])
        self.max_r = max([card.get_radius() for card in self.cards])

        self.bbox_spacing = self.max_width // 3
        self.bbox_size = self.max_width // 4
        self.bbox_half = self.bbox_size // 2

        self.card_bbox_x = [self.max_r, self.max_r + self.bbox_size]
        self.card_bbox_y = [self.max_r, self.max_r + self.bbox_size]

        self.rotate = iaa.Affine(rotate=(-180, 180))

        self.scene = self._generate_scene()

    def _generate_scene(self):
        # Bounding box of card center point placement
        # Best to base it off cards width
        no_cards = len(self.cards)
        scene_w = (
            (no_cards * (self.bbox_size))
            + ((no_cards - 1) * self.bbox_spacing)
            + 2 * self.max_r
        )
        scene_h = 2 * (self.max_r + self.bbox_half)

        scene_size = (scene_h, scene_w, 4)
        scene = np.zeros(scene_size, dtype=np.uint8)

        for card in self.cards:
            bounds = self._get_bounds()
            rotated = self._rotate(card)
            insert_position = self._get_card_insert_position(rotated, bounds)
            self._merge(scene, rotated, insert_position)

        return scene

    def _get_card_insert_position(self, card: Card, bounds) -> np.s_:
        rotated_w, rotated_h = card.shape[:2]

        offset_x = bounds[0] - rotated_w // 2
        offset_y = bounds[1] - rotated_h // 2

        return np.s_[offset_y : offset_y + rotated_h, offset_x : offset_x + rotated_w]

    # insert_position slice of background image on which to merge foreground image
    def _merge(self, background, foreground: np.array, insert_position: np.s_):
        roi = background[insert_position]

        foreground_gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(foreground_gray, 0, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        background_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        mask_fg = cv2.bitwise_and(foreground, foreground, mask=mask)

        dst = cv2.add(background_bg, mask_fg)
        background[insert_position] = dst

    def _rotate(self, card: Card) -> np.array:
        rotation_box = np.zeros((self.max_r * 2, self.max_r * 2, 4), dtype=np.uint8)
        card_w, card_h = card.get_size()

        offset_x = self.max_r - card_w // 2
        offset_y = self.max_r - card_h // 2

        rotation_box[
            offset_y : offset_y + card_h, offset_x : offset_x + card_w
        ] = card.image

        return self.rotate(image=rotation_box)

    def _get_bounds(self):
        _bounds = [
            np.random.randint(*self.card_bbox_x),
            np.random.randint(*self.card_bbox_y),
        ]
        self.card_bbox_x = [i + self.bbox_spacing for i in self.card_bbox_x]

        return _bounds

    def display(self):
        plt.axis("off")
        plt.imshow(self.scene)
        plt.show()

    def get_final(self):
        return self.final


if __name__ == "__main__":
    cf = CardFactory(TEST)
    bf = BackgroundFactory()

    bg = bf.get_random_background()
    cards = cf.get_random_cards(k=4)
    # print(card1, card2, sep="\n")

    scene = Scene(bg, cards)
    scene.display()
