from typing import List

import matplotlib.pyplot as plt
import numpy as np
from imgaug import ia
from imgaug import augmenters as iaa
from imgaug.augmentables import Polygon, PolygonsOnImage
from matplotlib import pyplot as plt
from PIL import Image

from background import Background
from card import Card
from card_factory import CardFactory
from config import (
    TEST,
    RETRIES,
    ROTATE_CARD,
    SCALE,
    SHEAR_X,
    SHEAR_Y,
)

SYMBOL_INTERSECT_RATIO = 0.5


class Scene:
    def __init__(self, cards: List[Card]) -> None:
        self.cards = cards
        self.scene = None
        self.scene_shape = None

        max_width = max([card.get_size()[0] for card in self.cards])
        self.max_r = max([card.get_radius() for card in self.cards])

        self.bbox_spacing = max_width // 3
        self.bbox_size = max_width // 4
        self.bbox_half = self.bbox_size // 2

        self.card_bbox_x = [self.max_r, self.max_r + self.bbox_size]
        self.card_bbox_y = [self.max_r, self.max_r + self.bbox_size]

        self.visible_polygons = []
        self.cards_polygons = []

        self.scene = self._generate_scene()
        self._transform_scene()

    def add_background(self, background: Background):
        bg_w, bg_h = background.get_size()
        scene_h, scene_w, _ = self.scene_shape

        if scene_h > scene_w:
            offset = (int((bg_w / 2) - (scene_w / 2)), 0)
            self._merge(background, self.scene, offset)
        else:
            offset = (0, int((bg_h / 2) - (scene_h / 2)))
            self._merge(background, self.scene, offset)

        self.visible_polygons.shift_(*offset)
        # ia.imshow(self._draw_polygons(np.array(background.get_image())))

    def get_visible_polygons(self):
        return self.visible_polygons

    def display(self):
        plt.axis("off")
        plt.imshow(self.scene)
        plt.show()

    # Get size in Pillow format
    def get_size(self):
        return (self.scene_shape[1], self.scene_shape[0])

    def _generate_scene(self):
        """
        Calculating scene width and height based: number of cards;
        size of bounding boxes (sqaure bounding box of user defined size in which card center is confined with
        each card having its own bounding box separated by some user defined spacing);
        size of bounding box spacing (space between bounding boxes);
        and max_r (half the radius of biggest card)
        """

        no_cards = len(self.cards)

        scene_w = (
            (no_cards * (self.bbox_size))
            + ((no_cards - 1) * self.bbox_spacing)
            + 2 * self.max_r
        )
        scene_h = 2 * (self.max_r + self.bbox_half)

        # Image in Pillow requires size to be in (width, height) format
        # whreas shape from image stored in numpy is (height, width, channels)
        # self.scene_shape is in numpy format
        self.scene_shape = (scene_h, scene_w, 4)
        scene = np.zeros(self.scene_shape, dtype=np.uint8)
        scene = Image.fromarray(scene)

        visible_polygons = {}

        for card in self.cards:
            card_center_point = self._get_card_center_point()
            rotated, anchor_point, visible_polygons = self._rotate(
                card, card_center_point, visible_polygons
            )

            if type(rotated) is not np.ndarray:
                continue

            self._merge(scene, rotated, anchor_point)

        # mapping all symbols polygons from dictionary into into PolygonsOnImage object
        self.visible_polygons = PolygonsOnImage(
            [
                Polygon.from_shapely(poly, label=card_value)
                for card_value, polygons in visible_polygons.items()
                for poly in polygons
            ],
            shape=self.scene_shape,
        )

        return scene

    def _get_card_center_point(self):
        center_point = [
            np.random.randint(*self.card_bbox_x),
            np.random.randint(*self.card_bbox_y),
        ]
        self.card_bbox_x = [i + self.bbox_spacing for i in self.card_bbox_x]

        return center_point

    def _rotate(self, card: Card, center_point, visible_polygons) -> np.array:
        card_image = np.array(card.get_image())
        card_h, card_w, _ = card_image.shape

        # Card figure translation and mapping to Polygons
        card_hulls_tranlated = [hull for hull in card._hulls]
        _card_symbols_polygons = [
            Polygon(hull, label=f"symbol {i}")
            for i, hull in enumerate(card_hulls_tranlated)
        ]

        # Card translation and mapping to Polygon
        _card_polygon = Polygon(
            [(0, 0), (card_w, 0), (card_w, card_h), (0, card_h)], label="card"
        )

        # Combining all card polygons into PolygonsOnImage
        card_polygons = PolygonsOnImage(
            [*_card_symbols_polygons, _card_polygon], shape=card_image.shape,
        )

        # trying n times till success
        rotate = iaa.Affine(rotate=ROTATE_CARD, fit_output=True)
        for _ in range(RETRIES):
            rotated_card, card_polygons_rotated = rotate(
                image=card_image, polygons=card_polygons
            )

            rotated_h, rotated_w, _ = rotated_card.shape
            # Card polygons translation on scene
            card_translation = (
                center_point[0] - rotated_w // 2,
                center_point[1] - rotated_h // 2,
            )

            # Shifting polygons to match their position on scene
            card_polygons_rotated.shift_(*card_translation)

            _card_symbols_polygons_rotated, _card_polygon_rotated = (
                card_polygons_rotated[:-1],
                card_polygons_rotated[-1],
            )

            _card_polygon_shapenly = Polygon.to_shapely_polygon(_card_polygon_rotated)
            updated_visible_polygons = {}

            for _card_class, _symbols_poly in visible_polygons.items():
                # Set overlapping flag to true, if added card doesn't overlap flag will be set to true
                overlapping = True
                updated_symbols_poly = []

                for _symbol_poly in _symbols_poly:
                    _symbol_poly_area = _symbol_poly.area

                    _poly_inter = _card_polygon_shapenly.intersection(_symbol_poly)

                    # Set flag when at least one is visble
                    if ((_symbol_poly_area - _poly_inter.area) / _symbol_poly_area) > (
                        1 - SYMBOL_INTERSECT_RATIO
                    ):
                        overlapping = False
                        updated_symbols_poly.append(_symbol_poly)

                updated_visible_polygons[_card_class] = updated_symbols_poly

                if overlapping:
                    break
            else:
                updated_visible_polygons[card.value] = [
                    Polygon.to_shapely_polygon(_symbol_polygon_rotated)
                    for _symbol_polygon_rotated in _card_symbols_polygons_rotated
                ]
                self.cards_polygons.append(_card_polygon_rotated)
                return rotated_card, card_translation, updated_visible_polygons

        return None, None, visible_polygons

    # Merging 2 pictures
    def _merge(self, background, foreground, anchor_point):
        if not isinstance(foreground, Image.Image):
            foreground = Image.fromarray(np.uint8(foreground))

        if isinstance(background, Background):
            background = background.get_image()

        background.paste(foreground, anchor_point, foreground)

    def _transform_scene(self):
        seq = iaa.Sequential(
            [
                iaa.Affine(scale=SCALE),
                iaa.ShearX(SHEAR_X, fit_output=True),
                iaa.ShearY(SHEAR_Y, fit_output=True),
            ]
        )
        scene, self.visible_polygons = seq(
            image=np.array(self.scene), polygons=self.visible_polygons
        )
        self.scene_shape = scene.shape
        self.scene = Image.fromarray(scene)


if __name__ == "__main__":
    cf = CardFactory(TEST)

    cards = cf.get_random_cards(k=2)
    scene = Scene(cards)
