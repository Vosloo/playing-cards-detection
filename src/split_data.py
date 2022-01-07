import shutil
from pathlib import Path
from random import shuffle

from config import (DATASET_LABELS_PATH, DATASET_IMAGES_PATH, TEST, TEST_SPLIT, TRAIN,
                    TRAIN_SPLIT, VAL)

labels_paths = Path(DATASET_LABELS_PATH).glob("**/*")
labels_files = sorted([x for x in labels_paths if x.is_file()])

images_paths = Path(DATASET_IMAGES_PATH).glob("**/*")
images_files = sorted([x for x in images_paths if x.is_file()])

files = list(zip(labels_files, images_files))
shuffle(files)

no_files = len(files)

train_size = int(TRAIN_SPLIT * no_files)
test_size = int((TRAIN_SPLIT + TEST_SPLIT) * no_files)

train, test, val = (
    files[:train_size],
    files[train_size:test_size],
    files[test_size:],
)

LABELS_TEST_PATH = DATASET_LABELS_PATH / TEST
LABELS_TRAIN_PATH = DATASET_LABELS_PATH / TRAIN
LABELS_VAL_PATH = DATASET_LABELS_PATH / VAL

IMAGES_TEST_PATH = DATASET_IMAGES_PATH / TEST
IMAGES_TRAIN_PATH = DATASET_IMAGES_PATH / TRAIN
IMAGES_VAL_PATH = DATASET_IMAGES_PATH / VAL

LABELS_TEST_PATH.mkdir(parents=True, exist_ok=True)
LABELS_TRAIN_PATH.mkdir(parents=True, exist_ok=True)
LABELS_VAL_PATH.mkdir(parents=True, exist_ok=True)

IMAGES_TEST_PATH.mkdir(parents=True, exist_ok=True)
IMAGES_TRAIN_PATH.mkdir(parents=True, exist_ok=True)
IMAGES_VAL_PATH.mkdir(parents=True, exist_ok=True)


for label, image in test:
    shutil.move(str(image), str(IMAGES_TEST_PATH))
    shutil.move(str(label), str(LABELS_TEST_PATH))

for label, image in train:
    shutil.move(str(image), str(IMAGES_TRAIN_PATH))
    shutil.move(str(label), str(LABELS_TRAIN_PATH))

for label, image in val:
    shutil.move(str(image), str(IMAGES_VAL_PATH))
    shutil.move(str(label), str(LABELS_VAL_PATH))
