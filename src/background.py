from PIL import Image

from config import DATASET_IMAGES_PATH, DATASET_LABELS_PATH


class Background:
    def __init__(self, fname) -> None:
        self._fname = fname
        self._image = self._load_img()

    def display(self):
        self._image.show()

    def get_size(self):
        return self._image.size

    def get_image(self) -> Image.Image:
        return self._image

    def get_fname(self):
        return self._fname

    def resize(self, new_size: tuple):
        self._image = self._image.resize(new_size)

    def save_background(self, fname, labels):
        self._image.save(DATASET_IMAGES_PATH / (fname + ".jpg"))

        with open(DATASET_LABELS_PATH / (fname + ".txt"), "w") as f:
            f.writelines(labels)

    @property
    def dtype(self):
        return self._image.dtype

    @property
    def shape(self):
        return self.get_size()

    def _load_img(self) -> Image.Image:
        return Image.open(self._fname)

    def __getitem__(self, item):
        return self._image[item]
