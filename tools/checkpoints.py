"""Atomic local assets and completed training-stage checkpoints."""
import json
import os
from pathlib import Path
import tempfile

import torch


def checkpoint_root(directory=None):
    return Path(directory) if directory is not None else Path(__file__).resolve().parents[1]/'checkpoints'


def atomic_write_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.'+path.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_model_checkpoint(model, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.'+path.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, stream)
        os.replace(temporary, path)
        atomic_write_bytes(path.with_suffix('.json'), (json.dumps(metadata, indent=2)+'\n').encode())
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print('CHECKPOINT_SAVED', str(path), flush=True)


def completed_checkpoint(path):
    return Path(path).is_file()


def load_model_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    print('CHECKPOINT_LOADED', str(path), flush=True)
