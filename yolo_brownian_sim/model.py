#!/usr/bin/env python3
"""
YOLOv3-Tiny 기반 레이저 중심점 검출 모델
간소화된 구조로 단일 레이저 스팟 감지에 최적화
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv + BatchNorm + LeakyReLU 블록"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class YOLOv3Tiny(nn.Module):
    """
    YOLOv3-Tiny 기반 단일 레이저 스팟 검출기
    
    출력: [batch, num_anchors * (5 + num_classes), H, W]
    - 5 = (x, y, w, h, objectness)
    - num_classes = 1 (레이저 스팟)
    """
    
    def __init__(self, input_channels=5, num_classes=1, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Backbone (특징 추출)
        self.layer1 = nn.Sequential(
            ConvBlock(input_channels, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            ConvBlock(16, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            ConvBlock(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            ConvBlock(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            ConvBlock(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        
        # Detection layers
        self.layer6 = nn.Sequential(
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0)
        )
        
        # 최종 출력 (anchor당 5 + num_classes)
        out_channels = num_anchors * (5 + num_classes)
        self.detect = nn.Conv2d(256, out_channels, 1, 1, 0)
    
    def forward(self, x):
        """
        입력: [batch, channels, H, W]
        출력: [batch, num_anchors * (5 + num_classes), H/32, W/32]
        """
        x = self.layer1(x)  # /2
        x = self.layer2(x)  # /4
        x = self.layer3(x)  # /8
        x = self.layer4(x)  # /16
        x = self.layer5(x)  # /32
        x = self.layer6(x)
        x = self.detect(x)
        return x


def decode_predictions(predictions, anchors, num_classes=1, conf_threshold=0.5):
    """
    YOLO 출력을 bounding box로 디코딩
    
    Args:
        predictions: [batch, num_anchors * (5 + num_classes), H, W]
        anchors: [(w1, h1), (w2, h2), ...] anchor 크기
        num_classes: 클래스 수
        conf_threshold: confidence 임계값
    
    Returns:
        boxes: [N, 4] (x_center, y_center, w, h) - 정규화된 좌표 (0-1)
        scores: [N] confidence 점수
    """
    batch_size, _, grid_h, grid_w = predictions.shape
    num_anchors = len(anchors)
    
    # Reshape: [batch, num_anchors, 5+num_classes, H, W]
    predictions = predictions.view(batch_size, num_anchors, 5 + num_classes, grid_h, grid_w)
    predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()  # [batch, anchors, H, W, 5+classes]
    
    # Grid 좌표 생성
    grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
    grid_y = grid_y.float().to(predictions.device)
    grid_x = grid_x.float().to(predictions.device)
    
    # 예측값 추출
    x = torch.sigmoid(predictions[..., 0])  # x offset
    y = torch.sigmoid(predictions[..., 1])  # y offset
    w = predictions[..., 2]  # width
    h = predictions[..., 3]  # height
    conf = torch.sigmoid(predictions[..., 4])  # objectness
    
    # 절대 좌표로 변환 (0-1 정규화)
    x_center = (x + grid_x) / grid_w
    y_center = (y + grid_y) / grid_h
    
    # Anchor 적용 (anchors는 0-1 정규화됨)
    anchors_tensor = torch.tensor(anchors, device=predictions.device).float()
    box_w = torch.exp(w) * anchors_tensor[:, 0].view(1, -1, 1, 1)
    box_h = torch.exp(h) * anchors_tensor[:, 1].view(1, -1, 1, 1)
    
    # Confidence 필터링
    mask = conf > conf_threshold
    
    # 결과 수집 (모든 batch에 대해 결과 반환, detection 없으면 빈 텐서)
    boxes_list = []
    scores_list = []
    
    for b in range(batch_size):
        valid_mask = mask[b]
        if valid_mask.sum() == 0:
            # Detection 없으면 빈 텐서 추가 (batch index 유지)
            boxes_list.append(torch.empty((0, 4), device=predictions.device))
            scores_list.append(torch.empty((0,), device=predictions.device))
        else:
            boxes = torch.stack([
                x_center[b][valid_mask],
                y_center[b][valid_mask],
                box_w[b][valid_mask],
                box_h[b][valid_mask]
            ], dim=1)
            
            scores = conf[b][valid_mask]
            
            boxes_list.append(boxes)
            scores_list.append(scores)
    
    return boxes_list, scores_list


def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression to remove duplicate detections
    
    Args:
        boxes: [N, 4] (x_center, y_center, w, h) - normalized coordinates (0-1)
        scores: [N] confidence scores
        iou_threshold: IoU threshold for NMS
    
    Returns:
        keep_indices: indices of boxes to keep after NMS
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    
    # Convert center format to corner format for IoU calculation
    x_center = boxes[:, 0]
    y_center = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    
    # Calculate areas
    areas = w * h
    
    # Sort by scores (descending)
    sorted_indices = torch.argsort(scores, descending=True)
    keep = []
    
    while len(sorted_indices) > 0:
        # Take the box with highest score
        current = sorted_indices[0]
        keep.append(current.item())
        
        if len(sorted_indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_x1 = x1[current]
        current_y1 = y1[current]
        current_x2 = x2[current]
        current_y2 = y2[current]
        current_area = areas[current]
        
        # Intersection
        inter_x1 = torch.max(current_x1, x1[sorted_indices[1:]])
        inter_y1 = torch.max(current_y1, y1[sorted_indices[1:]])
        inter_x2 = torch.min(current_x2, x2[sorted_indices[1:]])
        inter_y2 = torch.min(current_y2, y2[sorted_indices[1:]])
        
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        
        # Union
        other_areas = areas[sorted_indices[1:]]
        union_area = current_area + other_areas - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-6)
        
        # Keep boxes with IoU < threshold
        sorted_indices = sorted_indices[1:][iou < iou_threshold]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def get_laser_center(boxes, scores, roi_center=(0.5, 0.5), use_nms=True, iou_threshold=0.5):
    """
    검출된 bbox에서 레이저 중심점 추출
    
    Args:
        boxes: [N, 4] (x_center, y_center, w, h) - normalized coordinates (0-1)
        scores: [N] confidence
        roi_center: ROI 중심점 (정규화, 0-1) - 중심에 가까운 박스 우선 선택
        use_nms: NMS 적용 여부
        iou_threshold: NMS IoU threshold
    
    Returns:
        center: (x, y) 정규화된 중심 좌표 (0-1), None if no detection
    """
    if len(boxes) == 0:
        return None
    
    # NMS 적용
    if use_nms:
        keep_indices = nms(boxes, scores, iou_threshold=iou_threshold)
        if len(keep_indices) == 0:
            return None
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
    
    # ROI 중심과의 거리 계산 (가중치로 사용)
    roi_x, roi_y = roi_center
    box_centers_x = boxes[:, 0]
    box_centers_y = boxes[:, 1]
    distances = torch.sqrt((box_centers_x - roi_x)**2 + (box_centers_y - roi_y)**2)
    
    # 가중치: confidence * (1 - normalized_distance)
    # ROI 중심에 가까울수록 높은 가중치
    max_distance = torch.sqrt(torch.tensor(2.0))  # 대각선 길이 (0-1 좌표계)
    normalized_distances = distances / max_distance
    weights = scores * (1.0 - normalized_distances * 0.5)  # 거리 페널티 50%
    
    # 가장 높은 가중치를 가진 박스 선택
    best_idx = torch.argmax(weights)
    center_x = boxes[best_idx, 0].item()
    center_y = boxes[best_idx, 1].item()
    
    return (center_x, center_y)


def count_parameters(model):
    """모델 파라미터 수 계산"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


if __name__ == "__main__":
    print("🧪 Testing YOLOv3-Tiny Model")
    print("=" * 50)
    
    # 모델 생성
    model = YOLOv3Tiny(input_channels=5, num_classes=1, num_anchors=3)
    params = count_parameters(model)
    print(f"📊 Total parameters: {params['total']:,}")
    print(f"📊 Trainable parameters: {params['trainable']:,}")
    
    # 테스트 입력
    batch_size = 2
    test_input = torch.randn(batch_size, 5, 512, 512)
    
    # Forward pass
    print(f"\n✅ Input shape: {test_input.shape}")
    output = model(test_input)
    print(f"✅ Output shape: {output.shape}")
    
    # 예측 디코딩
    anchors = [(400/512, 400/512), (0.5, 0.5), (1.0, 1.0)]
    boxes_list, scores_list = decode_predictions(output, anchors, conf_threshold=0.1)
    
    print(f"\n📦 Batch 1: {len(boxes_list[0]) if boxes_list else 0} detections")
    if boxes_list and len(boxes_list[0]) > 0:
        center = get_laser_center(boxes_list[0], scores_list[0])
        print(f"🎯 Laser center: {center}")
    
    print("\n✅ Model test completed!")
