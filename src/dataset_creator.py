import json
import shutil
from copy import deepcopy
from pathlib import Path
from random import randint
from time import perf_counter

import imgaug as ia
import numpy as np
from tqdm import tqdm

from background_factory import BackgroundFactory
from card_factory import CardFactory
from config import (
    CLASS_MAPPING,
    DATA,
    DATASET_LABELS_PATH,
    IMAGES,
    RESIZE,
    ROOT_PATH,
    SCANS,
    DATASET_IMAGES_PATH,
    DATASET_LABELS_PATH
)
import config
from scene import Scene


class DatasetCreator:
    def __init__(self, max_no_cards, no_backgrounds, no_scenes) -> None:
        self.max_no_cards = max_no_cards
        self.no_backgrounds = no_backgrounds
        self.no_outputs = no_scenes
        self.labels_path = DATASET_LABELS_PATH
        self.images_path = DATASET_IMAGES_PATH

        self.card_factory = CardFactory(dataset_path=DATA)
        self.background_factory = BackgroundFactory()
        self.mapping = json.load(open(Path(ROOT_PATH, IMAGES, SCANS, CLASS_MAPPING)))

    def create_dataset(self) -> None:
        """
        Creates a dataset of scenes with cards and backgrounds where size of 
        the generated dataset is defined as no_backgrounds * no_scenes.
        """

        # Create dataset directory
        shutil.rmtree(self.labels_path, ignore_errors=True)
        shutil.rmtree(self.images_path, ignore_errors=True)
        self.labels_path.mkdir(parents=True, exist_ok=True)
        self.images_path.mkdir(parents=True, exist_ok=True)

        no_scenes = 0
        start = perf_counter()
        for _ in tqdm(range(self.no_outputs)):
            no_cards = randint(1, self.max_no_cards)

            cards = self.card_factory.get_random_cards(no_cards)
            backgrounds = self.background_factory.get_random_backgrounds(
                self.no_backgrounds
            )

            scene = Scene(cards)
            for background in backgrounds:
                background_copy = deepcopy(background)

                # Resize background to fit scene (to 1:1 ratio)
                max_size = max(scene.get_size())
                background_copy.resize((max_size, max_size))

                # TODO: Change name of background and method add_background
                # Add merge bg_cp and scene and return polygons
                polygons = scene.add_background(background_copy)

                # Scale background and update polygons
                background_copy.resize(RESIZE)

                # Convert polygons into bounding boxes
                bboxes = [
                    poly.project((max_size, max_size), RESIZE).to_bounding_box()
                    for poly in polygons
                ]

                labels = []
                size, _ = RESIZE
                for bbox in bboxes:
                    # Normalize dimmentions
                    yolo = list(
                        np.array(
                            [bbox.center_x, bbox.center_y, bbox.width, bbox.height]
                        )
                        / size
                    )
                    labels.append(
                        " ".join(str(i) for i in [self.mapping[bbox.label], *yolo])
                        + "\n"
                    )

                fname = f"{no_scenes}_" + "".join([card.get_fname() for card in cards])

                # Save image and labels
                background_copy.save_background(fname, labels)

                no_scenes += 1

        print("Time taken: {}".format(perf_counter() - start))
        print(f"Created {no_scenes} scenes")


def draw_polygons(polygons, image):
    ia.imshow(polygons.draw_on_image(image), backend="cv2")


if __name__ == "__main__":
    dc = DatasetCreator(max_no_cards=3, no_backgrounds=10, no_scenes=1000)
    dc.create_dataset()
