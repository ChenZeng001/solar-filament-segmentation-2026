"""Encode native instance label maps as the competition CSV."""
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as coco

from tools.data import make_input


def generate_instance_submission(predictor, data_root, output):
    """Run TEST inference and write compressed COCO RLE for every instance.

    predictor(name, channels) returns (native labels, instance areas, details).
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    folder = Path(data_root) / 'test/test_images'
    names = sorted(path.name for path in folder.glob('*.jpeg'))

    def read(name):
        image = cv2.imread(str(folder / name), cv2.IMREAD_GRAYSCALE)
        return make_input(image, 2048)

    rows, empty_images = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {i: pool.submit(read, names[i]) for i in range(min(2, len(names)))}
        for position, name in enumerate(names):
            channels = pending.pop(position).result()
            following = position + 2
            if following < len(names):
                pending[following] = pool.submit(read, names[following])
            labels, areas, _ = predictor(name, channels)
            if not len(areas):
                empty_images.append(name)
            for ident in range(1, len(areas) + 1):
                mask = np.asfortranarray(labels == ident, dtype=np.uint8)
                encoded = coco.encode(mask)
                rows.append((f'{Path(name).stem}_{ident}', encoded['counts'].decode('ascii')))
            print(f'Test inference: {position + 1}/{len(names)}', flush=True)
            del channels, labels, areas

    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, quoting=csv.QUOTE_NONE, lineterminator='\n')
        writer.writerow(['filament_id', 'segmentation_rle'])
        writer.writerows(rows)
    return dict(test_images=len(names), instances=len(rows), empty_images=empty_images, output=str(output))
