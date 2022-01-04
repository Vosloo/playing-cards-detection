from pathlib import Path

IMAGES = "images/"
BACKGROUNDS = "backgrounds/"
SCANS = "scans/"

DATASET = "dataset/"

DATA = "data/"
TEST = "test/"
TRAIN = "train/"
VAL = "val/"

ANNOTATIONS = "annotations/"
LABELS = "labels/"
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
SCALE =  (0.7, 1)
SHEAR_X = (-15, 15)
SHEAR_Y = (-15, 15)

RESIZE = (512, 512)
MAX_NO_CARDS = 3
NO_BACKGROUNDS = 10
NO_SCENES = 400

ROOT_PATH = Path(__file__).resolve().parents[1]
IMAGES_PATH = Path(ROOT_PATH, IMAGES, SCANS)

IMAGES_PATH = Path(ROOT_PATH, DATASET, IMAGES)
LABELS_PATH = Path(ROOT_PATH, DATASET, LABELS)

TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.2
