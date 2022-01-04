from background_factory import BackgroundFactory
from card_factory import CardFactory
from scene import Scene
from time import perf_counter
from config import DATASET, ROOT_PATH, IMAGES, ANNOTATIONS
from pathlib import Path


class DatasetCreator:
    def __init__(self, no_cards, no_backgrounds, no_outputs) -> None:
        self.no_cards = no_cards
        self.no_backgrounds = no_backgrounds
        self.no_outputs = no_outputs
        self.annotations_path = Path(ROOT_PATH, DATASET, ANNOTATIONS)
        self.images_path = Path(ROOT_PATH, DATASET, IMAGES)

        self.card_factory = CardFactory()
        self.background_factory = BackgroundFactory()

    def create_dataset(self) -> None:
        """
        Creates a dataset of scenes with cards and backgrounds where size of 
        the generated dataset is defined as no_backgrounds * no_outputs.
        """
        backgrounds = self.background_factory.get_random_backgrounds(
            self.no_backgrounds
        )

        # Create dataset directory
        self.annotations_path.mkdir(
            parents=True, exist_ok=True
        )
        self.images_path.mkdir(
            parents=True, exist_ok=True
        )

        no_scenes = 0
        start = perf_counter()
        for _ in range(self.no_outputs):
            cards = self.card_factory.get_random_cards(self.no_cards)
            scene = Scene(cards)
            for background in backgrounds:
                # Resize background to fit scene (and to 1:1 ratio)
                max_size = max(scene.get_size())
                background.resize((max_size, max_size))

                scene.add_background(background)
                
                # Scale background to 300x300
                background.resize((300, 300))
                # background.display()
                background.save_background(self.images_path.joinpath(f'img{no_scenes}.png'))
                
                # TODO: Add other transformations??
                # TODO: Fix Polygons
                # TODO: Change naming background -> canvas or smth
                # TODO: Rescale background
                # TODO: Save background
                no_scenes += 1

        print("Time taken: {}".format(perf_counter() - start))
        print(f"Created {no_scenes} scenes")


if __name__ == "__main__":
    dc = DatasetCreator(no_cards=2, no_backgrounds=1, no_outputs=1)
    dc.create_dataset()