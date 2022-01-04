from pathlib import Path

IMAGES = "images/"
BACKGROUNDS = "backgrounds/"
SCANS = "scans/"

DATASET = "dataset/"

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
HULL = "hull"
WIDTH = "width"
HEIGHT = "height"
CLASS = "class"

RETRIES = 3

ROTATE_CARD = (-180, 180)
SCALE =  (0.5, 1)
SHEAR_X = (-20, 20)
SHEAR_Y = (-20, 20)

RESIZE = (300, 300)

ROOT_PATH = Path(__file__).resolve().parents[1]
IMAGES_PATH = Path(ROOT_PATH, IMAGES, SCANS)
