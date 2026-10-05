"""
GFL (Generalized Focal Loss) builder, Image task only.

Paper: Li et al., "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object
       Detection" (NeurIPS 2020, arXiv 2006.04388). Origin of the Distribution Focal Loss (DFL) used by YOLOv6 / v8.
Code : official implus/GFocal is an mmdetection 2.x fork; the same classes in mmdetection 3.x are used:
       configs/gfl/gfl_r50_fpn_1x_coco.py, models/dense_heads/gfl_head.py (GFLHead, Integral),
       losses/gfocal_loss.py (QualityFocalLoss beta 2, DistributionFocalLoss 0.25), GIoULoss 2.0,
       task_modules/assigners/atss_assigner.py (ATSSAssigner topk 9), coders/distance_point_bbox_coder.py,
       AnchorGenerator(ratios [1], octave_base_scale 8, center_offset 0).
       Backbone / FPN shared with detectors/retinanet (ResNet-50, FPN P3-P7, extra convs on P5 output without
       ReLU = mmdet add_extra_convs='on_output'). ATSS assignment reuses detectors/atss (same algorithm as the
       mmdet ATSSAssigner).

    head : 4 x (3x3 conv + GN + ReLU) per branch, gfl_cls (C), gfl_reg (4 x 17 bins), learnable Scale per level
    box  : distances = Integral(softmax(17 bins)) x stride from the anchor centre
    loss : QFL (target = IoU of the decoded positive box) / #pos + GIoU and DFL weighted by max cls score
           / sum(weights)
Raw contract: sigmoid class probabilities (= QFL joint quality-class score), objectness = 1.

[Unified] as RetinaNet / FCOS / ATSS: no ImageNet weights, trainable BN, common init (head GN and prior bias
kept), no mean / std, 608 input, Adam 1e-4, common post-processing, PreciseBN.
"""


import torch
import torch.nn as nn

from detectors.atss.build import ATSSLoss

STRIDES = (8, 16, 32, 64, 128)


class ConvGN(nn.Module):
    """mmcv ConvModule(norm_cfg=GN(32), act ReLU): conv without bias (bias='auto' with a norm layer)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
        self.gn = nn.GroupNorm(32, cout)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activate(self.gn(self.conv(x)))


def distance2bbox(points, distance):
    return torch.stack([points[..., 0] - distance[..., 0], points[..., 1] - distance[..., 1],
                        points[..., 0] + distance[..., 2], points[..., 1] + distance[..., 3]], -1)


class _IdentityCoder:
    """lets ATSSLoss.prepare_targets return the matched GT boxes instead of encoded deltas"""

    def encode(self, gt_boxes, anchors):
        return gt_boxes


def mmdet_anchors(sizes_hw, device, octave_base_scale=8):
    """mmdet AnchorGenerator(ratios=[1], octave_base_scale=8, scales_per_octave=1, center_offset=0)"""
    out = []
    for (h, w), s in zip(sizes_hw, STRIDES):
        half = octave_base_scale * s / 2
        sy, sx = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32) * s,
                                torch.arange(w, device=device, dtype=torch.float32) * s, indexing="ij")
        sx, sy = sx.reshape(-1), sy.reshape(-1)
        out.append(torch.stack([sx - half, sy - half, sx + half, sy + half], 1))
    return out


def atss_assign(anchors_lv, targets, num_classes, topk=9):
    """mmdet ATSSAssigner -> labels [B, R] (num_classes = background), matched GT boxes [B, R, 4]"""
    a = ATSSLoss(num_classes, _IdentityCoder(), topk=topk)
    labels, boxes = a.prepare_targets(targets, anchors_lv)            # labels 1..C, 0 = background
    labels = torch.stack(labels) - 1
    labels[labels < 0] = num_classes
    return labels, torch.stack(boxes)
