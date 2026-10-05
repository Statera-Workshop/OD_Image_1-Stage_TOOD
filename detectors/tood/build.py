"""
TOOD (Task-aligned One-stage Object Detection) builder, Image task only.

Paper: Feng, Zhong, Gao, Scott, Huang, "TOOD: Task-aligned One-stage Object Detection" (ICCV 2021,
       arXiv 2108.07755). Origin of the Task-Aligned Assigner (TAL) used by YOLOv6 / v8 / PP-YOLOE.
Code : official fcjian/TOOD is an mmdetection 2.x fork; the same classes in mmdetection 3.x are used:
       configs/tood/tood_r50_fpn_1x_coco.py, models/dense_heads/tood_head.py (TOODHead, TaskDecomposition,
       deform_sampling, anchor_type 'anchor_free'), task_modules/assigners/task_aligned_assigner.py (topk 13,
       alpha 1, beta 6), initial ATSSAssigner (topk 9) + FocalLoss for the first 4 epochs (initial_epoch 4),
       then QualityFocalLoss on probabilities with normalised alignment metrics, GIoU 2.0, utils/misc.py
       SigmoidGeometricMean. mmcv deform_conv2d -> torchvision.ops.deform_conv2d. GFL / ATSS helpers from
       detectors/gfl and detectors/atss. The epoch comes from the common trainer's set_epoch hook.

[Unified] as RetinaNet / FCOS / ATSS / GFL: no ImageNet weights, trainable BN, common init (head GN, prior and
attention init kept where they are biases; conv weights replaced), no mean / std, 608 input, Adam 1e-4,
common post-processing, PreciseBN.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from detectors.anchor_free.common import HeadStats, split_targets, to_raw
from detectors.atss.build import Scale
from detectors.gfl.build import STRIDES, ConvGN, atss_assign, distance2bbox, mmdet_anchors
from detectors.retinanet.fpn import build_retinanet_fpn
from detectors.retinanet.resnet import build_resnet
from detectors.rtmdet.build import bbox_overlaps

INF = 100000000.0


class SigmoidGeometricMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        x_sigmoid, y_sigmoid = x.sigmoid(), y.sigmoid()
        z = (x_sigmoid * y_sigmoid).sqrt()
        ctx.save_for_backward(x_sigmoid, y_sigmoid, z)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x_sigmoid, y_sigmoid, z = ctx.saved_tensors
        grad_x = grad_output * z * (1 - x_sigmoid) / 2
        grad_y = grad_output * z * (1 - y_sigmoid) / 2
        return grad_x, grad_y


class TaskDecomposition(nn.Module):
    def __init__(self, feat_channels, stacked_convs, la_down_rate=8):
        super().__init__()
        self.feat_channels, self.stacked_convs = feat_channels, stacked_convs
        self.in_channels = feat_channels * stacked_convs
        self.layer_attention = nn.Sequential(nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(self.in_channels // la_down_rate, stacked_convs, 1),
                                             nn.Sigmoid())
        self.reduction_conv = ConvGN(self.in_channels, feat_channels)        # 1x1 in mmdet: weight reshaped below
        self.reduction_conv.conv = nn.Conv2d(self.in_channels, feat_channels, 1, bias=False)
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reduction_conv.conv.weight, 0, 0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.layer_attention(avg_feat)
        conv_weight = weight.reshape(b, 1, self.stacked_convs, 1) * \
            self.reduction_conv.conv.weight.reshape(1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels, self.in_channels)
        feat = torch.bmm(conv_weight, feat.reshape(b, self.in_channels, h * w)).reshape(b, self.feat_channels, h, w)
        return self.reduction_conv.activate(self.reduction_conv.gn(feat))


class TOODHead(nn.Module):
    def __init__(self, num_classes, in_channels=256, feat_channels=256, stacked_convs=6):
        super().__init__()
        self.inter_convs = nn.ModuleList([ConvGN(in_channels if i == 0 else feat_channels, feat_channels)
                                          for i in range(stacked_convs)])
        self.cls_decomp = TaskDecomposition(feat_channels, stacked_convs, stacked_convs * 8)
        self.reg_decomp = TaskDecomposition(feat_channels, stacked_convs, stacked_convs * 8)
        self.tood_cls = nn.Conv2d(feat_channels, num_classes, 3, padding=1)
        self.tood_reg = nn.Conv2d(feat_channels, 4, 3, padding=1)
        self.cls_prob_module = nn.Sequential(nn.Conv2d(feat_channels * stacked_convs, feat_channels // 4, 1),
                                             nn.ReLU(inplace=True), nn.Conv2d(feat_channels // 4, 1, 3, padding=1))
        self.reg_offset_module = nn.Sequential(nn.Conv2d(feat_channels * stacked_convs, feat_channels // 4, 1),
                                               nn.ReLU(inplace=True), nn.Conv2d(feat_channels // 4, 4 * 2, 3, padding=1))
        self.scales = nn.ModuleList([Scale(1.0) for _ in STRIDES])
        bias_cls = -math.log((1 - 0.01) / 0.01)
        for m in self.inter_convs:
            nn.init.normal_(m.conv.weight, 0, 0.01)
        for m in list(self.cls_prob_module) + [self.tood_cls, self.tood_reg]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        for m in self.reg_offset_module:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.cls_prob_module[-1].bias, bias_cls)
        nn.init.constant_(self.tood_cls.bias, bias_cls)

    @staticmethod
    def deform_sampling(feat, offset):
        b, c, h, w = feat.shape
        weight = feat.new_ones(c, 1, 1, 1)
        return deform_conv2d(feat, offset, weight, stride=1, padding=0, dilation=1)   # groups = c via weight

    def forward(self, feats, anchors_lv):
        cls_scores, bbox_preds = [], []
        for x, scale, stride, anchor in zip(feats, self.scales, STRIDES, anchors_lv):
            b, c, h, w = x.shape
            inter_feats = []
            for inter_conv in self.inter_convs:
                x = inter_conv(x)
                inter_feats.append(x)
            feat = torch.cat(inter_feats, 1)
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_feat, reg_feat = self.cls_decomp(feat, avg_feat), self.reg_decomp(feat, avg_feat)
            cls_score = SigmoidGeometricMean.apply(self.tood_cls(cls_feat), self.cls_prob_module(feat))
            reg_dist = scale(self.tood_reg(reg_feat).exp()).float().permute(0, 2, 3, 1).reshape(-1, 4)
            centers = ((anchor[:, :2] + anchor[:, 2:]) / 2 / stride).repeat(b, 1)
            reg_bbox = distance2bbox(centers, reg_dist).reshape(b, h, w, 4).permute(0, 3, 1, 2)
            bbox_pred = self.deform_sampling(reg_bbox.contiguous(), self.reg_offset_module(feat).contiguous())
            invalid = ((bbox_pred[:, [0]] > bbox_pred[:, [2]]) | (bbox_pred[:, [1]] > bbox_pred[:, [3]])).expand_as(
                bbox_pred)
            cls_scores.append(cls_score)
            bbox_preds.append(torch.where(invalid, reg_bbox, bbox_pred))       # in stride units
        return cls_scores, bbox_preds


class TaskAlignedAssigner:
    def __init__(self, topk=13):
        self.topk = topk

    @torch.no_grad()
    def assign(self, pred_scores, decode_bboxes, priors, gt_bboxes, gt_labels, alpha=1, beta=6):
        num_gt, num_bboxes = gt_bboxes.size(0), priors.size(0)
        assigned_gt_inds = priors.new_full((num_bboxes,), 0, dtype=torch.long)
        assign_metrics = priors.new_zeros((num_bboxes,))
        if num_gt == 0 or num_bboxes == 0:
            return assigned_gt_inds, priors.new_zeros((num_bboxes,)), assign_metrics
        overlaps = bbox_overlaps(decode_bboxes, gt_bboxes).detach()
        bbox_scores = pred_scores[:, gt_labels].detach()
        alignment_metrics = bbox_scores ** alpha * overlaps ** beta
        topk = min(self.topk, alignment_metrics.size(0))
        _, candidate_idxs = alignment_metrics.topk(topk, dim=0, largest=True)
        candidate_metrics = alignment_metrics[candidate_idxs, torch.arange(num_gt)]
        is_pos = candidate_metrics > 0
        priors_cx, priors_cy = (priors[:, 0] + priors[:, 2]) / 2.0, (priors[:, 1] + priors[:, 3]) / 2.0
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_priors_cx = priors_cx.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        ep_priors_cy = priors_cy.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)
        l_ = ep_priors_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_priors_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_priors_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_priors_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        is_pos = is_pos & is_in_gts
        overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        m = max_overlaps != -INF
        assigned_gt_inds[m] = argmax_overlaps[m] + 1
        assign_metrics[m] = alignment_metrics[m, argmax_overlaps[m]]
        return assigned_gt_inds, max_overlaps, assign_metrics


def focal_loss_with_prob(pred, target_labels, num_classes, gamma=2.0, alpha=0.25):
    """mmdet focal_loss.py_focal_loss_with_prob (activated=True), per element"""
    target = F.one_hot(target_labels, num_classes=num_classes + 1)[:, :num_classes].type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    return F.binary_cross_entropy(pred, target, reduction="none") * focal_weight


def quality_focal_loss_with_prob(pred_sigmoid, label, score, beta=2.0):
    """mmdet gfocal_loss.quality_focal_loss_with_prob (activated=True), per sample"""
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred_sigmoid.shape)
    loss = F.binary_cross_entropy(pred_sigmoid, zerolabel, reduction="none") * scale_factor.pow(beta)
    bg_class_ind = pred_sigmoid.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy(pred_sigmoid[pos, pos_label], score[pos],
                                                  reduction="none") * scale_factor.abs().pow(beta)
    return loss.sum(dim=1)


class TOOD(nn.Module):
    def __init__(self, num_classes, depth=50, initial_epoch=4):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = build_resnet(depth)
        self.fpn = build_retinanet_fpn(self.backbone.out_channels, 256, top_block_in="p5", relu_before_p7=False)
        self.head = TOODHead(num_classes)
        self.assigner = TaskAlignedAssigner(topk=13)
        self.initial_epoch, self.epoch = initial_epoch, 0
        self.precise_bn = True
        self.precise_bn_images = 1600
        self.precise_bn_batch_size = 16
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, x, targets=None):
        feats = self.fpn(self.backbone(x))
        anchors_lv = mmdet_anchors([f.shape[-2:] for f in feats], x.device)
        cls_scores, bbox_preds = self.head(feats, anchors_lv)
        B, C = x.shape[0], self.num_classes
        anchors = torch.cat(anchors_lv, 0)
        strides = torch.cat([torch.full((len(a),), s, device=x.device) for a, s in zip(anchors_lv, STRIDES)])
        cls = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1, C) for c in cls_scores], 1)           # probabilities
        boxes = torch.cat([b.permute(0, 2, 3, 1).reshape(B, -1, 4) for b in bbox_preds], 1) * strides[None, :, None]
        raw = to_raw(boxes, cls)
        if targets is None:
            return raw
        tgt = split_targets(targets, B, x.shape[2])
        R = anchors.shape[0]
        if self.epoch < self.initial_epoch:
            labels, gt_boxes = atss_assign(anchors_lv, tgt, C)
            metrics = (labels < C).float()                                    # bbox_weights[..., 0]
        else:
            labels = torch.full((B, R), C, dtype=torch.long, device=x.device)
            gt_boxes = torch.zeros(B, R, 4, device=x.device)
            metrics = torch.zeros(B, R, device=x.device)
            for b, t in enumerate(tgt):
                gi, ious, am = self.assigner.assign(cls[b].detach(), boxes[b].detach(), anchors, t["boxes"], t["labels"])
                pos = gi > 0
                labels[b, pos] = t["labels"][gi[pos] - 1]
                gt_boxes[b, pos] = t["boxes"][gi[pos] - 1]
                for g in torch.unique(gi[pos]):                              # normalised alignment metrics
                    idx = pos & (gi == g)
                    metrics[b, idx] = am[idx] / (am[idx].max() + 1e-7) * ious[idx].max()
        labels_f, metrics_f = labels.view(-1), metrics.view(-1)
        cls_f = cls.reshape(-1, C)
        if self.epoch < self.initial_epoch:
            loss_cls = focal_loss_with_prob(cls_f, labels_f, C).sum()
        else:
            loss_cls = quality_focal_loss_with_prob(cls_f, labels_f, metrics_f).sum()
        pos = (labels_f < C).nonzero().squeeze(1)
        if len(pos) > 0:
            st = strides.repeat(B)[pos]
            pred = boxes.reshape(-1, 4)[pos] / st[:, None]
            target = gt_boxes.view(-1, 4)[pos] / st[:, None]
            if self.epoch < self.initial_epoch:                                # centerness_target
                pa = anchors.repeat(B, 1)[pos]
                acx, acy = (pa[:, 0] + pa[:, 2]) / 2, (pa[:, 1] + pa[:, 3]) / 2
                gtb = gt_boxes.view(-1, 4)[pos]
                lr = torch.stack([acx - gtb[:, 0], gtb[:, 2] - acx], 1)
                tb = torch.stack([acy - gtb[:, 1], gtb[:, 3] - acy], 1)
                w = torch.sqrt((lr.min(1)[0] / lr.max(1)[0]) * (tb.min(1)[0] / tb.max(1)[0]))
            else:
                w = metrics_f[pos]
            giou = bbox_overlaps(pred, target, mode="giou", is_aligned=True)
            loss_bbox = 2.0 * ((1 - giou) * w).sum()
            bbox_avg = w.sum().clamp(min=1).item()
        else:
            loss_bbox, bbox_avg = boxes.sum() * 0, 1.0
        cls_avg = metrics_f.sum().clamp(min=1).item()
        loss_dict = {"loss_cls": loss_cls / cls_avg, "loss_bbox": loss_bbox / bbox_avg}
        loss_dict["losses"] = loss_dict["loss_cls"] + loss_dict["loss_bbox"]
        loss_dict["num_pos"] = torch.tensor(len(pos) / B)
        pos_mask = labels_f < C
        logits = torch.logit(cls_f.detach().clamp(1e-6, 1 - 1e-6))
        self.stats.update(loss_dict, cls_f.detach().max(-1).values, logits, pos_mask, labels_f[pos_mask])
        return loss_dict["losses"], raw

    def load_darknet_weights(self, path):
        raise NotImplementedError("TOOD has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    return TOOD(num_classes)
