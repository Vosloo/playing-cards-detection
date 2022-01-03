import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class Background:
    def __init__(self, fname) -> None:
        self._fname = fname
        self._image = self._load_img()
        self._size = self._image.shape

    def display(self):
        plt.axis("off")
        plt.imshow(self._image)

    def get_size(self):
        return self._size

    def get_fname(self):
        return self._fname

    def _load_img(self):
        return mpimg.imread(self._fname)
