"""Final subpixel mask features and all-query, all-expert utility supervision."""
import torch
from torch import nn
from tools.set_model import AreaLoss


def weighted_soft_class_loss(logits, targets, background_weight=.1):
    logs = logits.float().log_softmax(-1)
    weight = targets + background_weight * (1 - targets)
    return -(targets * logs[..., 0] + background_weight * (1 - targets) * logs[..., 1]).sum() / weight.sum().clamp_min(1e-12)


class SubpixelMaskFeatures(nn.Module):
    def __init__(self, channels=256, factor=4):
        super().__init__()
        self.factor = factor
        self.projection = nn.Conv2d(channels, channels * factor**2, 1)
        self.shuffle = nn.PixelShuffle(factor)
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.bias.zero_()
            weight = self.projection.weight.reshape(channels, factor**2, channels)
            for channel in range(channels):
                weight[channel, :, channel] = 1.

    def forward(self, value):
        return self.shuffle(self.projection(value))


def attach_native_projection(model):
    original = model.model.pixel_level_module.decoder.mask_projection
    bank = SubpixelMaskFeatures(original.out_channels).to(device=original.weight.device, dtype=original.weight.dtype)
    model.model.pixel_level_module.decoder.mask_projection = nn.Sequential(original, bank)
    model.criterion = ExpertUtilityLoss(model.config, model.weight_dict).to(original.weight.device)
    return model


@torch.no_grad()
def native_utility_targets(mask_logits, class_logits, originals, version_lengths):
    p = mask_logits.detach().float().sigmoid()
    raw = class_logits.detach().float().softmax(-1)[..., 0]
    q = .1 * raw / (1 - raw + .1 * raw)
    winner = (q[:, :, None, None] * p).argmax(1)
    ids = torch.arange(p.shape[1], device=p.device)[None, :, None, None]
    owned = (ids == winner[:, None]) & (p >= .5)
    values = []
    with torch.autocast(p.device.type, enabled=False):
        for index, original in enumerate(originals):
            prediction = owned[index].float().flatten(1)
            truth = original.float().flatten(1)
            intersection = prediction @ truth.T
            union = prediction.sum(1)[:, None] + truth.sum(1)[None] - intersection
            # Both inputs are native binary masks. Float32 integer sums are
            # exact within2048**2; compare integers before forming the ratio.
            hit = 2 * intersection > union
            iou = intersection.double() / union.double().clamp_min(1)
            numerator = (iou * hit).sum(1)
            responsibility = torch.zeros_like(numerator)
            cursor = 0
            for length in version_lengths[index]:
                counts = hit[:, cursor:cursor + length].sum(1)
                responsibility += counts + (counts == 0)
                cursor += length
            values.append((numerator / responsibility.clamp_min(1)).float())
    return torch.stack(values)


class ExpertUtilityLoss(AreaLoss):
    def set_quality_data(self, masks, lengths):
        self.originals, self.version_lengths = masks, lengths

    def loss_labels(self, class_queries_logits, class_labels, indices):
        return dict(loss_cross_entropy=weighted_soft_class_loss(class_queries_logits, self.current_utility))

    def forward(self, masks_queries_logits, class_queries_logits, mask_labels, class_labels,
                auxiliary_predictions=None):
        previous = getattr(self, 'current_utility', None)
        self.current_utility = native_utility_targets(masks_queries_logits, class_queries_logits,
                                                    self.originals, self.version_lengths)
        try:
            return super().forward(masks_queries_logits, class_queries_logits, mask_labels,
                                   class_labels, auxiliary_predictions)
        finally:
            self.current_utility = previous
