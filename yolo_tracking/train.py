#!/usr/bin/env python3
"""
YOLO Tracking 훈련 스크립트
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
sys.path.append('/hai/home/jdj/dvs')

from config import YOLOTrackingExperimentConfig, get_quick_test_config, get_standard_config
from dataset import create_train_val_loaders, load_frames_from_bin
from model import get_yolo_tracking_model, count_parameters

# YOLO Loss 재사용
try:
    from yolo_sim.train import YOLOLoss
except ImportError:
    class YOLOLoss(nn.Module):
        """간소화된 YOLO Loss"""
        def __init__(self, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.1):
            super().__init__()
            self.lambda_coord = lambda_coord
            self.lambda_obj = lambda_obj
            self.lambda_noobj = lambda_noobj
            self.mse = nn.MSELoss(reduction='sum')
            self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        
        def forward(self, predictions, targets, anchors):
            batch_size, _, grid_h, grid_w = predictions.shape
            num_anchors = len(anchors)
            
            pred = predictions.view(batch_size, num_anchors, 6, grid_h, grid_w)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
            grid_y = grid_y.float().to(predictions.device)
            grid_x = grid_x.float().to(predictions.device)
            
            target_x = targets[:, 0] * grid_w
            target_y = targets[:, 1] * grid_h
            grid_i = target_x.long().clamp(0, grid_w - 1)
            grid_j = target_y.long().clamp(0, grid_h - 1)
            
            total_loss = 0.0
            anchor_w, anchor_h = anchors[0]
            
            for b in range(batch_size):
                gi, gj = grid_i[b], grid_j[b]
                
                pred_xy = pred[b, 0, gj, gi, :2]
                target_xy = torch.tensor([target_x[b] - gi.float(), target_y[b] - gj.float()], 
                                        device=pred_xy.device)
                total_loss += self.mse(torch.sigmoid(pred_xy), target_xy)
                
                pred_wh = pred[b, 0, gj, gi, 2:4]
                target_wh_ratio = torch.tensor([targets[b, 2] / anchor_w, targets[b, 3] / anchor_h], 
                                              device=pred_wh.device)
                total_loss += self.mse(pred_wh, torch.log(target_wh_ratio + 1e-6))
                
                pred_conf = pred[b, 0, :, :, 4]
                total_loss += self.bce(pred_conf[gj, gi].unsqueeze(0), 
                                      torch.ones(1, device=predictions.device))
                
                noobj_mask = torch.ones(grid_h, grid_w, device=predictions.device, dtype=torch.bool)
                noobj_mask[gj, gi] = False
                if noobj_mask.sum() > 0:
                    total_loss += self.lambda_noobj * self.bce(
                        pred_conf[noobj_mask], 
                        torch.zeros(noobj_mask.sum(), device=predictions.device)
                    )
            
            return total_loss / batch_size


def train_yolo_tracking(config: YOLOTrackingExperimentConfig):
    """YOLO Tracking 훈련"""
    
    # Device
    if config.system.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.system.device)
    
    print(f"🔧 Using device: {device}")
    
    # 데이터 로딩
    frames = load_frames_from_bin(config.data.bin_file_path, config.data.max_frames)
    if len(frames) == 0:
        print("❌ No frames loaded!")
        return
    
    # 데이터로더
    train_loader, val_loader = create_train_val_loaders(
        frames, config.data, 
        train_ratio=config.data.train_ratio,
        batch_size=config.training.batch_size,
        num_workers=config.system.num_workers
    )
    
    # 모델
    model = get_yolo_tracking_model(
        input_channels=config.data.num_temporal_frames,
        num_classes=config.model.num_classes,
        num_anchors=config.model.num_anchors
    ).to(device)
    
    params = count_parameters(model)
    print(f"📊 Model: {config.model.model_name}")
    print(f"   Parameters: {params['total']:,}")
    
    # Loss, Optimizer
    criterion = YOLOLoss(
        lambda_coord=config.training.lambda_coord,
        lambda_obj=config.training.lambda_obj,
        lambda_noobj=config.training.lambda_noobj
    )
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate, 
                          weight_decay=config.training.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                     factor=config.training.scheduler_factor,
                                                     patience=config.training.scheduler_patience)
    
    anchors = config.model.anchors
    
    # 훈련 루프
    print(f"\n🚀 Starting training for {config.training.num_epochs} epochs")
    print("=" * 70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.training.num_epochs):
        # 훈련
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            total_loss, coord_loss, obj_loss = criterion(outputs, targets, anchors)
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        train_loss /= len(train_loader)
        
        # 검증
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                total_loss, _, _ = criterion(outputs, targets, anchors)
                val_loss += total_loss.item()
        
        val_loss /= len(val_loader)
        
        # 스케줄러
        scheduler.step(val_loss)
        
        # 출력
        print(f"Epoch {epoch+1}/{config.training.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Best 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(config.system.checkpoint_dir, 
                                    f"{config.experiment_name}_best.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print("  ⭐ Best model saved!")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.training.patience:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n✅ Training completed! Best val loss: {best_val_loss:.4f}")
    print(f"📁 Model saved to: {config.system.checkpoint_dir}/")


def main():
    print("🎯 YOLO Tracking Training")
    print("=" * 60)
    
    print("\n📝 Select configuration:")
    print("1. Quick test")
    print("2. Standard")
    
    try:
        choice = input("Enter choice (1-2, default=1): ").strip() or "1"
        config = get_quick_test_config() if choice == "1" else get_standard_config()
    except:
        print("Using quick test config")
        config = get_quick_test_config()
    
    print(f"\n✅ Config: {config.experiment_name}")
    print(f"   ROI: {config.data.roi_center}, size: {config.data.roi_size}")
    
    train_yolo_tracking(config)


if __name__ == "__main__":
    main()

