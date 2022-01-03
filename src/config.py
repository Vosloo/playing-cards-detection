from pathlib import Path

IMAGES = "images/"
BACKGROUNDS = "backgrounds/"
SCANS = "scans/"

DATA = "data/"
TEST = "test/"

ANNOTATIONS = "annotations/"
CONTOURS = "contours/"
INPUT = "input/"
LABELED = "labeled/"
OUTPUT = "output/"

CLASS_MAPPING = "class_mapping.json"

DTD = "dtd/"
BACKGROUNDS_PKL = "backgrounds.pkl"

BOUNDING_BOXES = "bounding_boxes"
HULLS = "hulls"
WIDTH = "width"
HEIGHT = "height"

ROOT_PATH = Path(__file__).resolve().parents[1]
IMAGES_PATH = Path(ROOT_PATH, IMAGES, SCANS)
