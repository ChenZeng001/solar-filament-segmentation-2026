"""Only the final expert dataset, fixed32+12 training loop and optimizer."""
import gc
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as coco

from tools.data import annotation_rle
from tools.checkpoints import atomic_write_bytes
from tools.reproducibility import configure_determinism


def seed_everything(seed=2026):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def seed_expert(seed, deterministic=False):
    seed_everything(seed)
    configure_determinism(deterministic)


def collate(batch):
    return torch.stack([image for image, _ in batch]), [target for _, target in batch]


def collate_single(batch):
    return batch[0]


def optimizer_for(model, lr):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        encoder = name.startswith('model.pixel_level_module.encoder.')
        decay = 0. if parameter.ndim <= 1 or 'embedder' in name else .05
        groups.setdefault((encoder, decay), []).append(parameter)
    return torch.optim.AdamW([dict(params=params, lr=lr * (.3 if encoder else 1.),
                 weight_decay=decay, base_lr=lr * (.3 if encoder else 1.))
                 for (encoder, decay), params in groups.items()])


def schedule(optimizer, step, total_steps):
    factor = (.025 + .975 * .5 * (1 + np.cos(np.pi * step / total_steps))) * min(1., (step + 1) / 300)
    for group in optimizer.param_groups:
        group['lr'] = group['base_lr'] * factor


class ExpertDataset(Dataset):
    def __init__(self, names, records, images, side, native=False):
        self.images, self.native, self.side = images, native, side
        self.rows = [(name, i) for name in names for i in range(len(records[name]))]
        self.rles = {name: [[annotation_rle(a, v['height'], v['width']) for a in v['annotations']]
                           for v in records[name]] for name in names}
        self.lengths = {name: [len(r) for r in self.rles[name]] for name in names}
        self.targets = {}
        if not native:
            def build(row):
                name, version = row
                masks = [cv2.resize(coco.decode(r).astype(np.float32), (side // 4, side // 4),
                                   interpolation=cv2.INTER_AREA).astype(np.float16)
                         for r in self.rles[name][version]]
                return row, np.stack(masks) if masks else np.empty((0, side//4, side//4), np.float16)
            with ThreadPoolExecutor(max_workers=4) as pool:
                self.targets.update(pool.map(build, self.rows))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        name, version = self.rows[index]
        image = self.images[name].astype(np.float32)
        if self.native:
            all_rles = [r for v in self.rles[name] for r in v]
            originals = coco.decode(all_rles).transpose(2, 0, 1) if all_rles else np.empty((0,2048,2048),np.uint8)
            first = sum(self.lengths[name][:version])
            masks = originals[first:first+self.lengths[name][version]]
        else:
            masks = self.targets[(name, version)]
            originals = None
        k, flip = random.randrange(4), random.random() < .5
        def augment(x):
            x = np.rot90(x, k, axes=(-2,-1))
            return x[...,::-1] if flip else x
        image, masks = augment(image), augment(masks)
        if originals is not None: originals = augment(originals)
        if random.random() < .5:
            image[0] = image[0] * random.uniform(.9,1.1) + random.uniform(-.15,.15)
            image[1:] *= random.uniform(.85,1.15)
        target = dict(masks=torch.from_numpy(masks.copy()), labels=torch.zeros(len(masks),dtype=torch.int64))
        if self.native:
            target.update(originals=torch.from_numpy(originals.copy()), lengths=self.lengths[name])
        return torch.from_numpy(image.copy()), target


def train_stage(model, dataset, seed, native=False, deterministic=False, history_path=None):
    device = next(model.parameters()).device
    seed_expert(seed, deterministic)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=1 if native else 4, shuffle=True,
                        num_workers=4, pin_memory=True, persistent_workers=False,
                        collate_fn=collate_single if native else collate, generator=generator)
    optimizer = optimizer_for(model, 1e-4 if native else 2e-4)
    epochs, accumulation = (12, 2) if native else (32, 1)
    total_steps = ((len(loader) + accumulation - 1) // accumulation) * epochs
    step, history = 0, []
    for epoch in range(epochs):
        model.train(); started = time.monotonic(); total = 0.
        for index, batch in enumerate(loader):
            if index % accumulation == 0:
                count = min(accumulation, len(loader) - index)
                schedule(optimizer, step, total_steps)
                optimizer.zero_grad(set_to_none=True)
            if native:
                image, target = batch
                values = image[None].to(device, non_blocking=True)
                masks = [target['masks'].to(device, non_blocking=True).float()]
                labels = [target['labels'].to(device, non_blocking=True)]
                original = [target['originals'].to(device, non_blocking=True)]
                model.criterion.set_quality_data(original, [target['lengths']])
            else:
                images, targets = batch
                values = images.to(device, non_blocking=True)
                masks = [t['masks'].to(device, non_blocking=True).float() for t in targets]
                labels = [t['labels'].to(device, non_blocking=True) for t in targets]
            with torch.autocast(device.type, dtype=torch.bfloat16):
                output = model(pixel_values=values, mask_labels=masks, class_labels=labels)
            loss = output.loss
            (loss / count).backward(); total += float(loss.detach())
            if native:
                model.criterion.set_quality_data([], [])
                del original
            del output, loss, values, masks, labels
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); step += 1
                if step in (1, 5) or step % (25 if native else 50) == 0:
                    print('EXPERT_PROGRESS', json.dumps(dict(stage='native' if native else 'warmup',
                          epoch=epoch + 1, optimizer_step=step, total_steps=total_steps,
                          epoch_seconds=time.monotonic() - started)), flush=True)
        row = dict(
            epoch=epoch + 1, epochs=epochs, loss=total / len(loader),
            optimizer_steps=step, seconds=time.monotonic() - started)
        history.append(row)
        if history_path is not None:
            atomic_write_bytes(history_path, (json.dumps(history, indent=2)+'\n').encode())
        print('EXPERT_NATIVE' if native else 'EXPERT_WARMUP', json.dumps(row), flush=True)
    model._expert_training_history = history
    model.zero_grad(set_to_none=True)
    del loader, optimizer
    gc.collect(); torch.cuda.empty_cache()
    return model
