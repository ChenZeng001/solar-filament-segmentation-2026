"""Native subpixel D4 fields; accept every nonempty learned-utility mask."""
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as coco
from scipy.optimize import linear_sum_assignment

VIEWS = [(k, reflected) for reflected in (False, True) for k in range(4)]


def transform(value, turns, reflected):
    value = torch.rot90(value, turns, (-2, -1))
    return value.flip(-1) if reflected else value


def undo(value, turns, reflected):
    if reflected:
        value = value.flip(-1)
    return torch.rot90(value, -turns, (-2, -1))


def encoded(hard_mask):
    # Transfer column-major uint8 data directly; the CPU transpose is a view.
    array = hard_mask.to(torch.uint8).t().contiguous().cpu().numpy().T
    return coco.encode(array)


@torch.no_grad()
def native_fields(output, turns, reflected):
    logits = output.masks_queries_logits[0].float()
    raw = output.class_queries_logits[0].float().softmax(-1)[:, 0]
    q = .1 * raw / (1 - raw + .1 * raw)
    p = F.interpolate(logits[:, None], size=(2048, 2048), mode='bilinear', align_corners=False)[:, 0].sigmoid()
    # Area is invariant to the coordinate transform; filter before making copies.
    keep = ((p >= .5).flatten(1).sum(1) >= 1) & (q > 0)
    p, q = p[keep], q[keep]
    return undo(p, turns, reflected).contiguous(), q


class OrbitAccumulator:
    def __init__(self, total_views=8, side=2048):
        self.total_views, self.side, self.seen = total_views, side, 0
        self.groups = []
        self.device = None

    @torch.no_grad()
    def add(self, probabilities, utilities):
        self.device = probabilities.device
        self.seen += 1
        eligible = ((probabilities >= .5).flatten(1).sum(1) >= 1) & (utilities > 0)
        probabilities, utilities = probabilities[eligible], utilities[eligible]
        if not len(probabilities):
            return
        qs = utilities.cpu().numpy().astype(np.float64)
        masks = [encoded(p >= .5) for p in probabilities]
        matched = set()
        if self.groups:
            iou = coco.iou([g['prototype'] for g in self.groups], masks, [0] * len(masks))
            prefix_q = np.asarray([g['utility_sum'] / max(1, self.seen - 1) for g in self.groups])
            affinity = np.where(iou > .5, iou * np.sqrt(prefix_q[:, None] * qs[None, :]), 0.)
            rows, columns = linear_sum_assignment(-affinity)
            for i, j in zip(rows, columns):
                if affinity[i, j] <= 0:
                    continue
                group = self.groups[int(i)]
                group['field'].add_(probabilities[int(j)], alpha=float(qs[j]))
                group['utility_sum'] += float(qs[j])
                # Match the float32 addition order of the pixel fields. In
                # particular, constant P=.5 must remain exactly on the boundary.
                group['normalizer'] = float(np.float32(group['normalizer']) + np.float32(qs[j]))
                group['observations'] += 1
                group['prototype'] = encoded(group['field'] / group['normalizer'] >= .5)
                matched.add(int(j))
        for j, q in enumerate(qs):
            if j not in matched:
                self.groups.append(dict(field=probabilities[j] * float(q), utility_sum=float(q), normalizer=float(q),
                                        observations=1, prototype=masks[j]))

    @torch.no_grad()
    def result(self):
        empty = lambda: (np.zeros((self.side, self.side), np.int32), np.empty(0, np.float64), [])
        selected = [g for g in self.groups if g['utility_sum'] / self.total_views > .225]
        if not selected:
            return empty()
        fields = torch.stack([g['field'] for g in selected])
        sums = torch.tensor([g['normalizer'] for g in selected], device=self.device, dtype=torch.float32)
        p = fields / sums[:, None, None]
        areas = (p >= .5).flatten(1).sum(1)
        eligible = areas >= 1
        fields, p, areas = fields[eligible], p[eligible], areas[eligible]
        selected = [g for g, keep in zip(selected, eligible.cpu().tolist()) if keep]
        if not selected:
            return empty()
        # Q*P == sum(q_v*p_v)/total_views, so the unnormalized fields give the same winner.
        winner = fields.argmax(0)
        labels = torch.zeros((self.side, self.side), dtype=torch.int32, device=self.device)
        kept, details = [], []
        for index, group in enumerate(selected):
            owned = (winner == index) & (p[index] >= .5)
            area = int(owned.sum())
            if area < 1:
                continue
            kept.append(area)
            labels[owned] = len(kept)
            details.append(dict(q=group['utility_sum'] / self.total_views,
                                view_matches=group['observations'], original_area=int(areas[index])))
        return labels.cpu().numpy(), np.asarray(kept, np.float64), details
