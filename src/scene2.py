import cv2
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt

from background import Background
from background_factory import BackgroundFactory
from card import Card
from card_factory import CardFactory
from config import TEST
from scene import Scene


class Scene2(Scene):
    def __init__(self, bg: Background, *cards: Card) -> None:
        super().__init__(bg, *cards)

        self.max_width = max([card.get_size()[0] for card in self.cards])
        self.max_r = max([card.get_radius() for card in self.cards])

        self.bbox_offset = self.max_width // 6
        self.bbox_size = self.max_width // 4
        self.bbox_half = self.bbox_size // 2

        self.scene = self._generate_scene()

    def _generate_scene(self):
        # Bounding box of card center point placement
        # Best to base it off cards width
        center_x = self.max_r + self.bbox_size + self.bbox_offset
        center_y = self.max_r + self.bbox_half

        scene_size = (center_y * 2, center_x * 2, 4)
        scene = np.zeros(scene_size, dtype=np.uint8)

        left_card, right_card = self.cards

        left_bounds = self._get_left_bounds(center_x, center_y)
        rotated = self._rotate(left_card)
        self._merge(scene, rotated, left_bounds)

        right_bounds = self._get_right_bounds(center_x, center_y)
        rotated = self._rotate(right_card)
        self._merge(scene, rotated, right_bounds)

        return scene

    def _merge(self, scene, rotated: np.array, bounds):
        rotated_w, rotated_h, _ = rotated.shape

        offset_x = bounds[0] - rotated_w // 2
        offset_y = bounds[1] - rotated_h // 2

        roi = scene[offset_y : offset_y + rotated_h, offset_x : offset_x + rotated_w]

        rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(rotated_gray, 0, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        scene_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        mask_fg = cv2.bitwise_and(rotated, rotated, mask=mask)

        dst = cv2.add(scene_bg, mask_fg)

        scene[offset_y : offset_y + rotated_h, offset_x : offset_x + rotated_w] = dst

    def _rotate(self, card: Card) -> np.array:
        rotation_box = np.zeros((self.max_r * 2, self.max_r * 2, 4), dtype=np.uint8)
        card_w, card_h = card.get_size()

        offset_x = self.max_r - card_w // 2
        offset_y = self.max_r - card_h // 2

        rotation_box[
            offset_y : offset_y + card_h, offset_x : offset_x + card_w
        ] = card.image

        rotate = iaa.Affine(rotate=(-180, 180))

        return rotate(image=rotation_box)

    def _get_left_bounds(self, center_x, center_y):
        left_bounds = [
            np.random.randint(
                center_x - self.bbox_size - self.bbox_offset,
                center_x - self.bbox_offset + 1,
            ),
            np.random.randint(center_y - self.bbox_half, center_y + self.bbox_half + 1),
        ]

        return left_bounds

    def _get_right_bounds(self, center_x, center_y):
        right_bounds = [
            np.random.randint(
                center_x + self.bbox_offset,
                center_x + self.bbox_size + self.bbox_offset + 1,
            ),
            np.random.randint(center_y - self.bbox_half, center_y + self.bbox_half + 1),
        ]

        return right_bounds


if __name__ == "__main__":
    cf = CardFactory(TEST)
    bf = BackgroundFactory()

    bg = bf.get_random_background()
    card1, card2 = cf.get_random_cards(k=2)
    print(card1, card2, sep="\n")

    scene = Scene2(bg, card1, card2)
    scene.display()
