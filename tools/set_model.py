"""Dense area-preserving set losses and exact COCO foreground marginalization."""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss, Mask2FormerHungarianMatcher

NATIVE_SIZE = 2048


def area_targets(masks, size):
    if not len(masks):
        return masks.new_empty((0, *size), dtype=torch.float32)
    if tuple(masks.shape[-2:]) == tuple(size):
        return masks.float()
    return F.interpolate(masks[:, None].float(), size=size, mode='area')[:, 0]


class ForegroundMarginal(torch.nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, x):
        logits = self.predictor(x).float()
        return torch.cat([torch.logsumexp(logits[..., :-1], dim=-1, keepdim=True), logits[..., -1:]], dim=-1)


class AreaMatcher(Mask2FormerHungarianMatcher):
    @torch.no_grad()
    def forward(self, masks_queries_logits, class_queries_logits, mask_labels, class_labels):
        result = []
        for logits, classes, truths, labels in zip(masks_queries_logits, class_queries_logits, mask_labels, class_labels):
            if not len(labels):
                result.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue
            h, w = logits.shape[-2:]
            target = area_targets(truths, (h, w)).flatten(1)
            with torch.autocast(logits.device.type, enabled=False):
                z = logits.float().flatten(1)
                p = z.sigmoid()
                ce = F.softplus(z).mean(1)[:, None] - (z @ target.T) / (h * w)
                epsilon = h * w / NATIVE_SIZE ** 2
                dice = 1 - (2 * (p @ target.T) + epsilon) / (p.sum(1)[:, None] + target.sum(1)[None] + epsilon)
                cost = self.cost_mask * ce + self.cost_dice * dice - self.cost_class * classes.float().softmax(-1)[:, labels]
            a, b = linear_sum_assignment(cost.cpu().numpy())
            result.append((torch.as_tensor(a, dtype=torch.int64), torch.as_tensor(b, dtype=torch.int64)))
        return result


class AreaLoss(Mask2FormerLoss):
    def __init__(self, config, weight_dict):
        super().__init__(config, weight_dict)
        self.matcher = AreaMatcher(cost_class=config.class_weight, cost_mask=config.mask_weight, cost_dice=config.dice_weight)

    def loss_labels(self, class_queries_logits, class_labels, indices):
        if not torch.are_deterministic_algorithms_enabled():
            return super().loss_labels(class_queries_logits, class_labels, indices)
        # CUDA's spatial NLL reduction is nondeterministic in the pinned torch
        # release. Express the same weighted mean without that reduction.
        logits = class_queries_logits.float()
        target = torch.full(logits.shape[:2], self.num_labels, dtype=torch.int64, device=logits.device)
        selected = self._get_predictions_permutation_indices(indices)
        target[selected] = torch.cat([labels[j] for labels, (_, j) in zip(class_labels, indices)])
        weights = self.empty_weight[target]
        logs = logits.log_softmax(-1).gather(-1, target[..., None])[..., 0]
        return dict(loss_cross_entropy=-(logs * weights).sum() / weights.sum())

    def loss_masks(self, masks_queries_logits, mask_labels, indices, num_masks):
        source = self._get_predictions_permutation_indices(indices)
        pred = masks_queries_logits[source].float()
        if not len(pred):
            zero = masks_queries_logits.sum() * 0
            return dict(loss_mask=zero, loss_dice=zero)
        h, w = pred.shape[-2:]
        target = torch.cat([area_targets(m, (h, w))[t] for m, (_, t) in zip(mask_labels, indices)])
        epsilon = h * w / NATIVE_SIZE ** 2
        with torch.autocast(pred.device.type, enabled=False):
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none').flatten(1).mean(1)
            p, t = pred.sigmoid().flatten(1), target.flatten(1)
            dice = 1 - (2 * (p * t).sum(1) + epsilon) / (p.sum(1) + t.sum(1) + epsilon)
        return dict(loss_mask=bce.sum() / num_masks, loss_dice=dice.sum() / num_masks)


def specialize(model):
    model.class_predictor = ForegroundMarginal(model.class_predictor)
    model.config.id2label = {0: 'filament'}
    model.config.label2id = {'filament': 0}
    model.criterion = AreaLoss(model.config, model.weight_dict)
    return model
