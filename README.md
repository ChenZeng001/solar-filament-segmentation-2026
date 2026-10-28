# Solar Filament Instance Segmentation

Final submission for the Solar Filament Segmentation Challenge 2026. The method uses a Swin-Tiny Mask2Former with native subpixel mask features, multi-expert quality supervision, and eight-view D4 inference.

**Public score: 0.41 · Submission: 57456520 · TEST images: 180 · Predicted instances: 1245.**

## Run the notebook

Use Linux, Python 3.12, and a CUDA GPU. The training environment used an NVIDIA RTX PRO 6000 Blackwell Server Edition with Torch 2.12.1+cu130 and CUDA 13.0.

```bash
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab main.ipynb
```

Place the official competition data in the following layout, or set `SOLAR_DATA_ROOT` to its location:

```text
MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── MAGFiLO_1.0_Annotations_kaggle2026_train.json
│   └── train_images/*.jpeg
└── test/
    └── test_images/*.jpeg
```

Set `DEVICE` and the data path in the notebook's configuration cell, then run the cells in order. The default device is `cuda`. The notebook loads the supplied weights and writes `results.csv`. TEST images are sufficient for inference; TRAIN images and annotations are used when training.

## Workflow

`main.ipynb` presents the complete workflow:

1. Load the official images and expert annotation versions.
2. Build brightness and two-scale absorption input channels.
3. Construct the pretrained Mask2Former and dense set losses.
4. Load completed stages or run the 32-epoch warmup and 12-epoch native training.
5. Associate instances across eight D4 views and fuse quality-weighted probability fields.
6. Assign instance pixels and export compressed COCO RLE.
7. Display a generated prediction.

The reference prediction is included at `reference/submission.csv`.

## Method

The three input channels combine grayscale intensity with two relative dark-absorption contrasts. The Swin-Tiny backbone supplies multiscale features to a Mask2Former with 100 instance queries. Its COCO foreground logits are combined with logsumexp into one filament class.

Each expert annotation version forms a training sample. Hungarian matching and dense BCE/Dice supervise the selected expert's contours. Native-resolution classification targets aggregate IoU hits and false-positive responsibility across all experts for that image.

| Stage | Epochs | Input / mask size | Batch / accumulation | Learning rate |
|---|---:|---|---|---|
| Warmup | 32 | 1024 / 256 | 4 / 1 | 2e-4 |
| Native | 12 | 2048 / 2048 | 1 / 2 | 1e-4 |

The native stage adds a 256→4096 pointwise projection and PixelShuffle(4), producing full-resolution mask features. Both stages use AdamW, an encoder learning-rate multiplier of 0.3, a 300-update warmup, cosine decay, gradient clipping at 1, and seed 16220.

Inference associates instances across eight ordered D4 views using IoU and learned quality. Groups with quality > 0.225 enter pixel competition; the winning group's probability must be >= 0.5 for a pixel to belong to that instance.

To train from the generic pretrained model, set `TRAIN_FROM_SCRATCH=True`. New stage weights are saved under `checkpoints/retrained/`. With the default setting, the notebook reuses the supplied weights under `checkpoints/expert_set_deterministic/`. Generic assets are loaded locally, with downloading used for missing files.

## Repository layout

```text
main.ipynb                              Complete workflow
requirements.txt                        Pinned Python dependencies
pyproject.toml                          Package metadata
tools/                                  Model, data, training and inference helpers
checkpoints/
  pretrained/mask2former_swin_tiny_coco/
    config.json
    model.safetensors
  expert_set_deterministic/
    warmup_final.pt
    warmup_final.json
    native_final.pt
    native_final.json
reference/submission.csv                 Submitted prediction
```

## Upload to GitHub

The weight files use Git LFS through `.gitattributes`. Install Git LFS and initialize it before adding the files:

```bash
git init
git lfs install
git add .
git lfs ls-files
git commit -m "Add final solar filament submission"
```

Set the GitHub remote and push the repository together with its LFS objects.

Pretrained model: [facebook/mask2former-swin-tiny-coco-instance](https://huggingface.co/facebook/mask2former-swin-tiny-coco-instance/tree/22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6).
