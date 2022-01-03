from typing import List, overload

import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables import Keypoint, Polygon, KeypointsOnImage, PolygonsOnImage
from matplotlib import pyplot as plt

from background import Background
from background_factory import BackgroundFactory

from card import Card
from card_factory import CardFactory

from config import TEST
import sys
from time import perf_counter

import matplotlib.pyplot as plt

CARD_POLY_LABEL = "card"
SYMBOL_INTERSECT_RATIO = 0.5


class Scene:
    def __init__(self, bg: Background, cards: List[Card]) -> None:
        self.bg = bg
        self.cards = cards
        self.scene = None
        self.scene_shape = None

        self.max_width = max([card.get_size()[0] for card in self.cards])
        self.max_r = max([card.get_radius() for card in self.cards])

        self.bbox_spacing = self.max_width // 3
        self.bbox_size = self.max_width // 4
        self.bbox_half = self.bbox_size // 2

        self.card_bbox_x = [self.max_r, self.max_r + self.bbox_size]
        self.card_bbox_y = [self.max_r, self.max_r + self.bbox_size]

        self.rotate = iaa.Affine(rotate=(-180, 180))
        self.card_visibility = {}

        self.scene = self._generate_scene()

    def _generate_scene(self):
        no_cards = len(self.cards)
        scene_w = (
            (no_cards * (self.bbox_size))
            + ((no_cards - 1) * self.bbox_spacing)
            + 2 * self.max_r
        )
        scene_h = 2 * (self.max_r + self.bbox_half)

        self.scene_shape = (scene_h, scene_w, 4)
        scene = np.zeros(self.scene_shape, dtype=np.uint8)

        for card in self.cards:
            card_center_point = self._get_card_center_point()
            rotated = self._rotate(card, card_center_point)

            if type(rotated) is not np.ndarray:
                print("Failed to add card to scene. Skipping card")
                continue

            insert_position = self._get_card_insert_position(rotated, card_center_point)
            self._merge(scene, rotated, insert_position)

        return scene

    def _get_card_insert_position(self, rotated: np.array, bounds: List[int]) -> np.s_:
        rotated_w, rotated_h = rotated.shape[:2]

        offset_x = bounds[0] - rotated_w // 2
        offset_y = bounds[1] - rotated_h // 2

        return np.s_[offset_y : offset_y + rotated_h, offset_x : offset_x + rotated_w]

    def _rotate(self, card: Card, center_point) -> np.array:
        rotation_box = np.zeros((self.max_r * 2, self.max_r * 2, 4), dtype=np.uint8)
        card_w, card_h = card.get_size()

        card_half_w, card_half_h = card_w // 2, card_h // 2

        offset_x = self.max_r - card_half_w
        offset_y = self.max_r - card_half_h

        rotation_box[
            offset_y : offset_y + card_h, offset_x : offset_x + card_w
        ] = card.image

        # card translation from scene [0,0]
        card_translation = (
            center_point[0] - card_half_w,
            center_point[1] - card_half_h,
        )
        translate = iaa.Affine(translate_px=card_translation)

        # create card polygons
        # Don't change order
        card_polygons = PolygonsOnImage(
            [
                *[
                    Polygon(hull, label=f"symbol_{i}")
                    for i, hull in enumerate(card.hulls)
                ],
                Polygon(
                    [(0, 0), (card_w, 0), (card_w, card_h), (0, card_h)],
                    label=CARD_POLY_LABEL,
                ),
            ],
            shape=self.scene_shape,
        )

        # translate card polygons (polygons of the symbols and polygon of card itself)
        card_polygons = translate(polygons=card_polygons)

        # trying n times till success
        for i in range(3):
            rotated_image, card_polygons_rotated = self.rotate(
                image=rotation_box, polygons=card_polygons
            )
            card_symbols_polygons, card_polygon = (
                card_polygons_rotated[:-1],
                card_polygons_rotated[-1],
            )

            card_polygon_shapenly = Polygon.to_shapely_polygon(card_polygon)

            for _card_polygons in self.card_visibility.values():
                # Set overlapping flag to true, if added card doesn't overlap flag will be set to true
                overlapping = True

                for _symbol_polygon in _card_polygons:
                    _symbol_polygon_area = _symbol_polygon.area

                    poly_intersection = card_polygon_shapenly.intersection(
                        _symbol_polygon
                    )

                    # when at least one symbol is visible
                    if (
                        (_symbol_polygon_area - poly_intersection.area)
                        / _symbol_polygon_area
                    ) > SYMBOL_INTERSECT_RATIO:
                        overlapping = False
                        break

                if overlapping:
                    break
            else:
                self.card_visibility[card.value] = [
                    Polygon.to_shapely_polygon(card_symbol_polygon)
                    for card_symbol_polygon in card_symbols_polygons
                ]
                return rotated_image

        return None

    def _get_card_center_point(self):
        center_point = [
            np.random.randint(*self.card_bbox_x),
            np.random.randint(*self.card_bbox_y),
        ]
        self.card_bbox_x = [i + self.bbox_spacing for i in self.card_bbox_x]

        return center_point

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
    scene = Scene(bg, cards)
    scene.display()
