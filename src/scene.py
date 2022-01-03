from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables import Polygon, PolygonsOnImage
from matplotlib import pyplot as plt
from PIL import Image

from background import Background
from card import Card
from card_factory import CardFactory
from config import RETRIES, TEST

SYMBOL_INTERSECT_RATIO = 0.5


class Scene:
    def __init__(self, cards: List[Card]) -> None:
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

    def add_background(self, background: Background):
        _, bg_h = background.get_size()
        _, scene_h = self.scene_shape
        offset_y = int((bg_h / 2) - (scene_h / 2))

        self._merge(
            background,
            self.scene,
            (0, offset_y),
        )

    def display(self):
        plt.axis("off")
        plt.imshow(self.scene)
        plt.show()

    def get_size(self):
        return self.scene_shape

    def _generate_scene(self):
        no_cards = len(self.cards)
        # Calculating scene width and height based:
        # - number of cards
        # - size of bounding boxes (sqaure bounding box of user defined size in which card center is confined. Each card has its own bounding box
        # separated by some user defined spacing)
        # - size of bounding box spacing (space between bounding boxes)
        # - max_r (half the radius of biggest card)
        scene_w = (
            (no_cards * (self.bbox_size))
            + ((no_cards - 1) * self.bbox_spacing)
            + 2 * self.max_r
        )
        scene_h = 2 * (self.max_r + self.bbox_half)

        self.scene_shape = (scene_w, scene_h)

        scene = Image.new('RGBA', self.scene_shape)

        for card in self.cards:
            card_center_point = self._get_card_center_point()
            rotated = self._rotate(card, card_center_point)

            if type(rotated) is not np.ndarray:
                print("Failed to add card to scene. Skipping card")
                continue

            anchor_point = self._get_card_anchor_point(rotated, card_center_point)
            self._merge(scene, rotated, anchor_point)

        return scene

    def _get_card_center_point(self):
        center_point = [
            np.random.randint(*self.card_bbox_x),
            np.random.randint(*self.card_bbox_y),
        ]
        self.card_bbox_x = [i + self.bbox_spacing for i in self.card_bbox_x]

        return center_point

    def _rotate(self, card: Card, center_point) -> np.array:
        rotation_box = np.zeros((self.max_r * 2, self.max_r * 2, 4), dtype=np.uint8)
        card_w, card_h = card.get_size()

        card_half_w, card_half_h = card_w // 2, card_h // 2

        offset_x = self.max_r - card_half_w
        offset_y = self.max_r - card_half_h

        rotation_box[
            offset_y : offset_y + card_h, offset_x : offset_x + card_w
        ] = card.numpy_image()

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
                    for i, hull in enumerate(card._hulls)
                ],
                Polygon(
                    [(0, 0), (card_w, 0), (card_w, card_h), (0, card_h)], label="card",
                ),
            ],
            shape=self.scene_shape,
        )

        # translate card polygons (polygons of the symbols and polygon of card itself)
        card_polygons = translate(polygons=card_polygons)

        # trying n times till success
        for _ in range(RETRIES):
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

    def _get_card_anchor_point(
        self, rotated: np.array, bounds: List[int]
    ) -> Tuple[int]:
        rot_w, rot_h, _ = rotated.shape
        offset_x = bounds[0] - rot_w // 2
        offset_y = bounds[1] - rot_h // 2

        return (offset_x, offset_y)

    # Merging 2 pictures
    def _merge(self, background, foreground, anchor_point):
        if not isinstance(foreground, Image.Image):
            foreground = Image.fromarray(np.uint8(foreground))

        if isinstance(background, Background):
            background = background.get_image()

        background.paste(foreground, anchor_point, foreground)


if __name__ == "__main__":
    cf = CardFactory(TEST)

    cards = cf.get_random_cards(k=3)
    scene = Scene(cards)
    scene.display()
