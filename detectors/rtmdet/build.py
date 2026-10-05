"""
RTMDet builder (RTMDet-s by default; Image task only).

Paper: Lyu et al., "RTMDet: An Empirical Study of Designing Real-Time Object Detectors" (arXiv 2212.07784, 2022).
Code : official open-mmlab/mmdetection 3.x (configs/rtmdet/rtmdet_s_8xb32-300e_coco.py, rtmdet_l base), classes
       extracted without the mmengine / mmcv registry:
       models/layers/csp_layer.py            CSPNeXtBlock, ChannelAttention, CSPLayer
       models/backbones/cspnext.py           CSPNeXt (P5), csp_darknet.py SPPBottleneck (5, 9, 13)
       models/necks/cspnext_pafpn.py         CSPNeXtPAFPN
       models/dense_heads/rtmdet_head.py     RTMDetSepBNHead (conv weights shared across levels, separate BN,
                                             pred kernel 1, exp_on_reg False for s / tiny), RTMDetHead loss
       task_modules/assigners/dynamic_soft_label_assigner.py   DynamicSoftLabelAssigner (topk 13, iou_weight 3,
                                             soft_center_radius 3)
       losses/gfocal_loss.py (QualityFocalLoss beta 2, weight 1), iou_loss.py (GIoULoss weight 2)
       prior_generators MlvlPointGenerator (offset 0, strides 8 / 16 / 32), coders distance2bbox
       mmcv ConvModule -> Conv2d(bias=False) + BN + SiLU; DepthwiseSeparableConvModule -> dw + pw ConvModule.

[Unified] SyncBN -> BN (single GPU; SyncBN defaults momentum 0.1, eps 1e-5 kept); no ImageNet CSPNeXt weights;
common init (conv weights / BN; head prior bias kept); no mean / std; Mosaic / MixUp / large-scale jitter
replaced by the common augmentation; 608 input; Adam 1e-4 instead of AdamW 0.004 + flat-cosine + EMA; common
post-processing (instead of score 0.001, top-1000, NMS 0.65, max 300). PreciseBN (BN everywhere, micro batch 2).
"""


import torch


EPS, INF = 1.0e-7, 100000000.0


def bbox_overlaps(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6):
    """mmdet structures/bbox/bbox_overlaps.py (2-D case)"""
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    if is_aligned:
        lt, rb = torch.max(bboxes1[..., :2], bboxes2[..., :2]), torch.min(bboxes1[..., 2:], bboxes2[..., 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        union = area1 + area2 - overlap
        enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
        enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        union = area1[..., None] + area2[..., None, :] - overlap
        enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
    union = torch.max(union, union.new_tensor([eps]))
    ious = overlap / union
    if mode == "iou":
        return ious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = torch.max(enclose_wh[..., 0] * enclose_wh[..., 1], union.new_tensor([eps]))
    return ious - (enclose_area - union) / enclose_area
