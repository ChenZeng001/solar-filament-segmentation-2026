"""Official data, expert annotations and the final absorption representation."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

import cv2
import numpy as np
from pycocotools import mask as coco_mask

cv2.setNumThreads(1)


def load_catalog(root):
    root = Path(root)
    source = root / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    data = json.loads(source.read_text())
    annotations = defaultdict(list)
    for annotation in data['annotations']:
        annotations[annotation['image_id']].append(annotation)
    records = defaultdict(list)
    for image in data['images']:
        records[image['file_name']].append({**image, 'annotations': annotations[image['id']]})
    records = dict(sorted(records.items()))
    return records


def annotation_rle(annotation, height=2048, width=2048):
    segmentation = annotation['segmentation']
    if isinstance(segmentation, dict):
        return coco_mask.frPyObjects(segmentation, height, width) if isinstance(segmentation['counts'], list) else segmentation
    return coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))


def make_input(image, size):
    """Separate brightness and absorption at two spatial scales."""
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    background = cv2.GaussianBlur(image, (0, 0), 16 * size / 1024)
    broad = cv2.GaussianBlur(image, (0, 0), 64 * size / 1024)
    valid = broad > 0.12
    contrast = np.clip((background - image) / np.maximum(background, .12), -.3, .3)
    broad_contrast = np.clip((broad - image) / np.maximum(broad, .12), -.4, .4)
    channels = np.stack([image, .5 - 2 * contrast, .5 - 1.5 * broad_contrast])
    channels[1:, ~valid] = .5
    channels = (channels - np.array([.485, .456, .406], np.float32)[:, None, None]) / np.array([.229, .224, .225], np.float32)[:, None, None]
    return channels.astype(np.float16)


def preload_images(root, names, size, workers=8):
    folder = Path(root) / 'train/train_images'
    def read(name):
        image = cv2.imread(str(folder / name), cv2.IMREAD_GRAYSCALE)
        return name, make_input(image, size)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(read, names))
