from typing import List

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
        self.visible_polygons = {}

        self.scene = self._generate_scene()

    def add_background(self, background: Background):
        _, bg_h = background.get_size()
        _, scene_h = self.scene_shape
        offset_y = int((bg_h / 2) - (scene_h / 2))

        self._merge(
            background, self.scene, (0, offset_y),
        )

    def display(self):
        plt.axis("off")
        plt.imshow(self.scene)
        plt.show()

    def get_size(self):
        return self.scene_shape

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

        self.scene_shape = (scene_w, scene_h)

        scene = Image.new("RGBA", self.scene_shape)

        for card in self.cards:
            card_center_point = self._get_card_center_point()
            rotated, anchor_point = self._rotate(card, card_center_point)

            if type(rotated) is not np.ndarray:
                print("Failed to add card to scene. Skipping card")
                continue

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
        # Create rotation layer to preserve card shape while rotating
        rotation_layer = np.zeros((self.max_r * 2, self.max_r * 2, 4), dtype=np.uint8)
        card_w, card_h = card.get_size()

        card_half_w, card_half_h = card_w // 2, card_h // 2

        offset_x = self.max_r - card_half_w
        offset_y = self.max_r - card_half_h

        rotation_layer[
            offset_y : offset_y + card_h, offset_x : offset_x + card_w
        ] = card.numpy_image()

        # Card figure translation and mapping to Polygons
        card_hulls_tranlated = [hull for hull in card._hulls]
        _card_symbols_polygons = [
            Polygon(hull, label=f"symbol_{i}")
            for i, hull in enumerate(card_hulls_tranlated)
        ]

        # Card translation and mapping to Polygon
        _card_polygon = Polygon(
            [(0, 0), (card_w, 0), (card_w, card_h), (0, card_h)], label="card"
        )

        # Combining all card polygons into PolygonsOnImage
        card_polygons = PolygonsOnImage(
            [*_card_symbols_polygons, _card_polygon], shape=rotation_layer.shape,
        )

        # Card polygons translation on rotation layer
        first_card_translation = [
            offset_x,
            offset_y,
        ]

        card_polygons = card_polygons.shift(*first_card_translation)

        # Card polygons translation on scene
        second_card_translation = (
            center_point[0] - self.max_r,
            center_point[1] - self.max_r,
        )

        # trying n times till success
        for _ in range(RETRIES):
            rotated_image, card_polygons_rotated = self.rotate(
                image=rotation_layer, polygons=card_polygons
            )

            card_polygons_rotated = card_polygons_rotated.shift(
                *second_card_translation
            )

            _card_symbols_polygons_rotated, _card_polygon_rotated = (
                card_polygons_rotated[:-1],
                card_polygons_rotated[-1],
            )

            _card_polygon_shapenly = Polygon.to_shapely_polygon(_card_polygon_rotated)
            updated_visible_polygons = {}

            for _card_class, _symbols_poly in self.visible_polygons.items():
                # Set overlapping flag to true, if added card doesn't overlap flag will be set to true
                overlapping = True
                updated_symbols_poly = []

                for _symbol_poly in _symbols_poly:
                    _symbol_poly_area = _symbol_poly.area

                    _poly_inter = _card_polygon_shapenly.intersection(_symbol_poly)

                    # Set flag when at least one is visble
                    if (
                        (_symbol_poly_area - _poly_inter.area) / _symbol_poly_area
                    ) > SYMBOL_INTERSECT_RATIO:
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
                self.visible_polygons = updated_visible_polygons
                return rotated_image, second_card_translation

        return None, None

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
    # scene.display()
