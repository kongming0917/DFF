#!/usr/bin/env python3
"""YOLOv3-Tiny 단일 레이저 스팟 검출기 + dvslib 루프용 어댑터.

옛 yolo_brownian_sim의 model.py(YOLOv3Tiny·decode·NMS·center)와 train.py(YOLOLoss)를 그대로 옮기고,
dvslib 공유 루프가 요구하는 두 hook을 붙였다:
  - YOLOCriterion : (out, xy 타깃) → loss. bbox 타깃은 중심 좌표 + 고정 크기(laser_diameter)로 생성
                    → 데이터셋은 CNN과 동일한 DVSBrownianDataset을 그대로 쓴다.
  - YOLOCenterDecoder : out → (B,2) 중심 좌표. decode → NMS → ROI 중심 우선 선택.
                    검출 실패 시 직전 성공 좌표 유지(초기 (0.5,0.5)) — 옛 inference 의미 그대로.
입력 스케일: 옛 코드는 프레임을 max(=2)로 나눠 0/0.5/1로 넣었다. 동일 동작을 모델 안의
`input_scale=0.5`로 흡수해 데이터 파이프라인(raw 0/1/2)은 CNN과 공유한다.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

LASER_DIAMETER = 400          # 레이저 스팟 직경(px) — bbox 크기·anchor 기준
DEFAULT_ROI = 512


def default_anchors(roi: int = DEFAULT_ROI) -> List[Tuple[float, float]]:
    laser = LASER_DIAMETER / roi
    return [(laser, laser), (0.5, 0.5), (1.0, 1.0)]


class ConvBlock(nn.Module):
    """Conv + BatchNorm + LeakyReLU 블록"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class YOLOv3Tiny(nn.Module):
    """출력: [batch, num_anchors * (5 + num_classes), H/32, W/32]  (5 = x, y, w, h, objectness)"""

    def __init__(self, input_channels=5, num_classes=1, num_anchors=3, input_scale: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.input_scale = input_scale  # raw 0/1/2 → 0/0.5/1 (옛 /max 정규화와 동일). buffer가 아니라 state_dict 호환.

        self.layer1 = nn.Sequential(ConvBlock(input_channels, 16, 3, 1, 1), nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(ConvBlock(16, 32, 3, 1, 1), nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(ConvBlock(32, 64, 3, 1, 1), nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(ConvBlock(64, 128, 3, 1, 1), nn.MaxPool2d(2, 2))
        self.layer5 = nn.Sequential(ConvBlock(128, 256, 3, 1, 1), nn.MaxPool2d(2, 2))
        self.layer6 = nn.Sequential(ConvBlock(256, 512, 3, 1, 1), ConvBlock(512, 256, 1, 1, 0))
        self.detect = nn.Conv2d(256, num_anchors * (5 + num_classes), 1, 1, 0)

    def forward(self, x):
        x = x * self.input_scale
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return self.detect(x)


def decode_predictions(predictions, anchors, num_classes=1, conf_threshold=0.5):
    """YOLO 출력 → batch별 (boxes [N,4] (xc, yc, w, h) 정규화, scores [N])."""
    batch_size, _, grid_h, grid_w = predictions.shape
    num_anchors = len(anchors)
    predictions = predictions.view(batch_size, num_anchors, 5 + num_classes, grid_h, grid_w)
    predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()

    grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
    grid_y = grid_y.float().to(predictions.device)
    grid_x = grid_x.float().to(predictions.device)

    x = torch.sigmoid(predictions[..., 0])
    y = torch.sigmoid(predictions[..., 1])
    w = predictions[..., 2]
    h = predictions[..., 3]
    conf = torch.sigmoid(predictions[..., 4])

    x_center = (x + grid_x) / grid_w
    y_center = (y + grid_y) / grid_h
    anchors_tensor = torch.tensor(anchors, device=predictions.device).float()
    box_w = torch.exp(w) * anchors_tensor[:, 0].view(1, -1, 1, 1)
    box_h = torch.exp(h) * anchors_tensor[:, 1].view(1, -1, 1, 1)
    mask = conf > conf_threshold

    boxes_list, scores_list = [], []
    for b in range(batch_size):
        valid = mask[b]
        if valid.sum() == 0:
            boxes_list.append(torch.empty((0, 4), device=predictions.device))
            scores_list.append(torch.empty((0,), device=predictions.device))
        else:
            boxes_list.append(torch.stack(
                [x_center[b][valid], y_center[b][valid], box_w[b][valid], box_h[b][valid]], dim=1))
            scores_list.append(conf[b][valid])
    return boxes_list, scores_list


def nms(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    areas = boxes[:, 2] * boxes[:, 3]
    order = torch.argsort(scores, descending=True)
    keep = []
    while len(order) > 0:
        cur = order[0]
        keep.append(cur.item())
        if len(order) == 1:
            break
        rest = order[1:]
        inter_w = torch.clamp(torch.min(x2[cur], x2[rest]) - torch.max(x1[cur], x1[rest]), min=0)
        inter_h = torch.clamp(torch.min(y2[cur], y2[rest]) - torch.max(y1[cur], y1[rest]), min=0)
        inter = inter_w * inter_h
        iou = inter / (areas[cur] + areas[rest] - inter + 1e-6)
        order = rest[iou < iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def get_laser_center(boxes, scores, roi_center=(0.5, 0.5), use_nms=True, iou_threshold=0.5):
    """검출 bbox 중 confidence·ROI 중심 근접도 가중치가 가장 큰 박스의 중심 (x, y). 없으면 None."""
    if len(boxes) == 0:
        return None
    if use_nms:
        keep = nms(boxes, scores, iou_threshold=iou_threshold)
        if len(keep) == 0:
            return None
        boxes, scores = boxes[keep], scores[keep]
    roi_x, roi_y = roi_center
    distances = torch.sqrt((boxes[:, 0] - roi_x) ** 2 + (boxes[:, 1] - roi_y) ** 2)
    weights = scores * (1.0 - distances / torch.sqrt(torch.tensor(2.0)) * 0.5)
    best = torch.argmax(weights)
    return (boxes[best, 0].item(), boxes[best, 1].item())


class YOLOLoss(nn.Module):
    """단일 객체용 간소화 YOLO loss: CIoU(첫 anchor) + objectness BCE (positive 1 cell, negative 가중 λ)."""

    def __init__(self, anchors, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.1):
        super().__init__()
        self.anchors = anchors
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")

    @staticmethod
    def _ciou_loss(p, t):
        px1, py1, px2, py2 = p[0] - p[2] / 2, p[1] - p[3] / 2, p[0] + p[2] / 2, p[1] + p[3] / 2
        tx1, ty1, tx2, ty2 = t[0] - t[2] / 2, t[1] - t[3] / 2, t[0] + t[2] / 2, t[1] + t[3] / 2
        inter = (torch.clamp(torch.min(px2, tx2) - torch.max(px1, tx1), min=0)
                 * torch.clamp(torch.min(py2, ty2) - torch.max(py1, ty1), min=0))
        union = p[2] * p[3] + t[2] * t[3] - inter + 1e-6
        iou = inter / union
        center_d2 = (p[0] - t[0]) ** 2 + (p[1] - t[1]) ** 2
        enc_d2 = ((torch.max(px2, tx2) - torch.min(px1, tx1)) ** 2
                  + (torch.max(py2, ty2) - torch.min(py1, ty1)) ** 2 + 1e-6)
        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(t[2] / (t[3] + 1e-6)) - torch.atan(p[2] / (p[3] + 1e-6)), 2)
        alpha = v / (1 - iou + v + 1e-6)
        return 1 - (iou - center_d2 / enc_d2 - alpha * v)

    def forward(self, predictions, targets):
        """predictions [B, A*6, H, W], targets [B, 4+] (xc, yc, w, h, ...) 정규화."""
        batch_size, _, grid_h, grid_w = predictions.shape
        num_anchors = len(self.anchors)
        pred = predictions.view(batch_size, num_anchors, 6, grid_h, grid_w).permute(0, 1, 3, 4, 2).contiguous()
        gi = (targets[:, 0] * grid_w).long().clamp(0, grid_w - 1)
        gj = (targets[:, 1] * grid_h).long().clamp(0, grid_h - 1)
        anchor_w, anchor_h = self.anchors[0]
        dev = predictions.device

        coord_loss = torch.zeros((), device=dev)
        obj_loss = torch.zeros((), device=dev)
        for b in range(batch_size):
            i, j = gi[b], gj[b]
            xy = torch.sigmoid(pred[b, 0, j, i, :2])
            wh = pred[b, 0, j, i, 2:4]
            pred_box = torch.stack([(xy[0] + i.float()) / grid_w, (xy[1] + j.float()) / grid_h,
                                    torch.exp(wh[0]) * anchor_w, torch.exp(wh[1]) * anchor_h])
            coord_loss = coord_loss + self._ciou_loss(pred_box, targets[b, :4])

            conf = pred[b, 0, :, :, 4]
            obj_loss = obj_loss + self.bce(conf[j, i].unsqueeze(0), torch.ones(1, device=dev))
            mask = torch.ones(grid_h, grid_w, device=dev, dtype=torch.bool)
            mask[j, i] = False
            obj_loss = obj_loss + self.lambda_noobj * self.bce(conf[mask], torch.zeros(int(mask.sum()), device=dev))
        return (self.lambda_coord * coord_loss + obj_loss) / batch_size


class YOLOCriterion(nn.Module):
    """dvslib 루프 hook: (out, xy 타깃 [B,2]) → loss. bbox 타깃 = 중심 + 고정 크기."""

    def __init__(self, anchors, roi: Tuple[int, int] = (DEFAULT_ROI, DEFAULT_ROI),
                 laser_diameter: int = LASER_DIAMETER, **loss_kwargs):
        super().__init__()
        self.loss = YOLOLoss(anchors, **loss_kwargs)
        roi_h, roi_w = roi
        self.wh = (min(1.0, laser_diameter / roi_w), min(1.0, laser_diameter / roi_h))

    def forward(self, out, xy):
        b = xy.shape[0]
        wh = torch.tensor(self.wh, device=xy.device, dtype=xy.dtype).expand(b, 2)
        return self.loss(out, torch.cat([xy[:, :2].clamp(0, 1), wh], dim=1))


class YOLOCenterDecoder:
    """dvslib 루프 hook: out → (B,2) 중심 좌표. 검출 실패 시 직전 성공 좌표 유지 (상태, reset()으로 초기화)."""

    def __init__(self, anchors, conf_threshold: float = 0.6, roi_center=(0.5, 0.5)):
        self.anchors = anchors
        self.conf_threshold = conf_threshold
        self.roi_center = roi_center
        self.reset()

    def reset(self):
        self.last = self.roi_center
        self.n_total = 0
        self.n_detect = 0

    @property
    def detection_rate(self) -> float:
        return 100.0 * self.n_detect / max(1, self.n_total)

    @torch.no_grad()
    def __call__(self, out: torch.Tensor) -> torch.Tensor:
        boxes_list, scores_list = decode_predictions(out, self.anchors, conf_threshold=self.conf_threshold)
        centers = []
        for boxes, scores in zip(boxes_list, scores_list):
            c = get_laser_center(boxes, scores, roi_center=self.roi_center) if len(boxes) > 0 else None
            if c is not None:
                self.last = (float(c[0]), float(c[1]))
                self.n_detect += 1
            self.n_total += 1
            centers.append(self.last)
        return torch.tensor(centers, device=out.device, dtype=torch.float32)


def get_model(model_name: str = "yolo_tiny", input_channels: int = 5, **kwargs) -> nn.Module:
    if model_name != "yolo_tiny":
        raise ValueError(f"Unknown model: {model_name}. Available: ['yolo_tiny']")
    return YOLOv3Tiny(input_channels=input_channels, num_classes=1, num_anchors=3, **kwargs)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


if __name__ == "__main__":
    m = get_model()
    print("params:", count_parameters(m))
    x = torch.randint(0, 3, (2, 5, 512, 512)).float()
    out = m(x)
    print("out:", tuple(out.shape))
    dec = YOLOCenterDecoder(default_anchors(), conf_threshold=0.1)
    print("centers:", dec(out).tolist(), "detect%:", dec.detection_rate)
    crit = YOLOCriterion(default_anchors())
    print("loss:", crit(out, torch.tensor([[0.5, 0.5], [0.3, 0.7]])).item())
