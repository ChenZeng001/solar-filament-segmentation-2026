"""Load local COCO pretraining assets or download the fixed model revision."""
import json
import re
import subprocess
import urllib.error
import urllib.request

import torch
from safetensors.torch import load
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

from tools.checkpoints import checkpoint_root, atomic_write_bytes

REVISION = '22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6'
BASE = 'https://huggingface.co/facebook/mask2former-swin-tiny-coco-instance/resolve/' + REVISION
ASSET_NAMES = ('config.json', 'model.safetensors')


def download_bytes(url):
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return subprocess.check_output([
            'curl', '--disable', '--ipv4', '--http1.1', '--fail', '--location',
            '--silent', '--show-error', '--connect-timeout', '20',
            '--max-time', '180', url,
        ])


def read_assets(checkpoints_dir=None):
    folder = checkpoint_root(checkpoints_dir) / 'pretrained/mask2former_swin_tiny_coco'
    assets = {}
    for name in ASSET_NAMES:
        path = folder / name
        if path.is_file():
            print('PRETRAINED_LOCAL', str(path), flush=True)
            assets[name] = path.read_bytes()
        else:
            print('PRETRAINED_DOWNLOAD', name, flush=True)
            assets[name] = download_bytes(BASE + '/' + name)
            atomic_write_bytes(path, assets[name])
    return assets


def load_pretrained_model(assets):
    """Adapt the Swin state layout to the installed Transformers model."""
    config = Mask2FormerConfig.from_dict(json.loads(assets['config.json']))
    state = load(assets['model.safetensors'])
    index_pattern = re.compile(
        r'^model\.pixel_level_module\.encoder\.encoder\.layers\.\d+\.blocks\.\d+'
        r'\.attention\.self\.relative_position_index$'
    )
    for name in list(state):
        if index_pattern.fullmatch(name):
            del state[name]

    # The segmentation backbone reads stage-normalized features. Its final
    # LayerNorm uses the library's unit/zero initialization.
    backbone = config.backbone_config
    channels = backbone.embed_dim * 2 ** (len(backbone.depths) - 1)
    prefix = 'model.pixel_level_module.encoder.swin.layernorm.'
    state.setdefault(prefix + 'weight', torch.ones(channels))
    state.setdefault(prefix + 'bias', torch.zeros(channels))
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        None, config=config, state_dict=state
    )
    print('PRETRAINED_LOADED Swin-Tiny', flush=True)
    return model
