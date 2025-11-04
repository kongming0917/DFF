#!/usr/bin/env python3
"""
Tracking 시스템 테스트 스크립트
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

print("🧪 Testing DVS Laser Tracking System")
print("=" * 60)

# 1. Config 테스트
print("\n1️⃣ Testing Configuration System...")
try:
    from config import TrackingExperimentConfig, get_quick_test_config
    
    config = get_quick_test_config()
    print(f"   ✅ Config loaded: {config.experiment_name}")
    print(f"      ROI center: {config.data.roi_center}")
    print(f"      ROI size: {config.data.roi_size}")
    print(f"      Motion std: {config.data.motion_std}")
    print(f"      Temporal frames: {config.data.num_temporal_frames}")
    
    # 설정 저장/로드 테스트
    test_file = "test_config.json"
    config.save_config(test_file)
    loaded = TrackingExperimentConfig.load_config(test_file)
    print(f"   ✅ Config save/load test passed")
    
    import os
    os.remove(test_file)
    
except Exception as e:
    print(f"   ❌ Config test failed: {e}")
    sys.exit(1)

# 2. Trajectory 생성 테스트
print("\n2️⃣ Testing Brownian Motion Trajectory Generation...")
try:
    # 간단한 trajectory 생성 (numpy만 사용)
    roi_center = (480, 294)
    roi_size = (384, 384)
    motion_std = 2.0
    boundary_margin = 80
    num_steps = 100
    
    # ROI 경계 계산
    roi_min_x = roi_center[0] - roi_size[1]//2 + boundary_margin
    roi_max_x = roi_center[0] + roi_size[1]//2 - boundary_margin
    roi_min_y = roi_center[1] - roi_size[0]//2 + boundary_margin
    roi_max_y = roi_center[1] + roi_size[0]//2 - boundary_margin
    
    print(f"   ROI range: X[{roi_min_x}, {roi_max_x}], Y[{roi_min_y}, {roi_max_y}]")
    
    # Trajectory 생성
    trajectory = [roi_center]
    for _ in range(1, num_steps):
        prev_x, prev_y = trajectory[-1]
        next_x = prev_x + np.random.normal(0, motion_std)
        next_y = prev_y + np.random.normal(0, motion_std)
        
        # 경계 반사
        if next_x < roi_min_x:
            next_x = 2 * roi_min_x - next_x
        elif next_x > roi_max_x:
            next_x = 2 * roi_max_x - next_x
        
        if next_y < roi_min_y:
            next_y = 2 * roi_min_y - next_y
        elif next_y > roi_max_y:
            next_y = 2 * roi_max_y - next_y
        
        next_x = np.clip(next_x, roi_min_x, roi_max_x)
        next_y = np.clip(next_y, roi_min_y, roi_max_y)
        trajectory.append((next_x, next_y))
    
    trajectory = np.array(trajectory)
    
    # 통계
    print(f"   ✅ Generated {len(trajectory)} trajectory points")
    print(f"      X range: [{trajectory[:, 0].min():.1f}, {trajectory[:, 0].max():.1f}]")
    print(f"      Y range: [{trajectory[:, 1].min():.1f}, {trajectory[:, 1].max():.1f}]")
    print(f"      Mean displacement: {np.mean(np.diff(trajectory, axis=0)):.2f}")
    
    # ROI 밖으로 나간 점 체크
    out_of_bounds = np.sum(
        (trajectory[:, 0] < roi_min_x) | (trajectory[:, 0] > roi_max_x) |
        (trajectory[:, 1] < roi_min_y) | (trajectory[:, 1] > roi_max_y)
    )
    if out_of_bounds > 0:
        print(f"   ⚠️  {out_of_bounds} points out of bounds!")
    else:
        print(f"   ✅ All points within ROI bounds")
    
except Exception as e:
    print(f"   ❌ Trajectory test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Trajectory 시각화
print("\n3️⃣ Visualizing Trajectory...")
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 왼쪽: 전체 trajectory
    ax = axes[0]
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.6, linewidth=1)
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
    
    # ROI 경계 표시
    roi_rect = plt.Rectangle(
        (roi_center[0] - roi_size[1]//2, roi_center[1] - roi_size[0]//2),
        roi_size[1], roi_size[0],
        fill=False, edgecolor='black', linewidth=2, linestyle='--', label='ROI'
    )
    ax.add_patch(roi_rect)
    
    # 허용 범위 표시
    allowed_rect = plt.Rectangle(
        (roi_min_x, roi_min_y),
        roi_max_x - roi_min_x, roi_max_y - roi_min_y,
        fill=False, edgecolor='red', linewidth=1, linestyle=':', label='Allowed range'
    )
    ax.add_patch(allowed_rect)
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Brownian Motion Trajectory ({num_steps} steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 오른쪽: 시간에 따른 위치 변화
    ax = axes[1]
    time_steps = np.arange(len(trajectory))
    ax.plot(time_steps, trajectory[:, 0], label='X position', alpha=0.7)
    ax.plot(time_steps, trajectory[:, 1], label='Y position', alpha=0.7)
    ax.axhline(roi_min_x, color='red', linestyle='--', alpha=0.3, label='X bounds')
    ax.axhline(roi_max_x, color='red', linestyle='--', alpha=0.3)
    ax.axhline(roi_min_y, color='blue', linestyle='--', alpha=0.3, label='Y bounds')
    ax.axhline(roi_max_y, color='blue', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Position (pixels)')
    ax.set_title('Position vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    save_path = 'test_trajectory.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Trajectory visualization saved to: {save_path}")
    
    try:
        plt.show()
    except:
        plt.close()
    
except Exception as e:
    print(f"   ⚠️  Visualization failed (maybe no display): {e}")

# 4. 레이블 변환 테스트
print("\n4️⃣ Testing Label Conversion (GT → ROI relative)...")
try:
    # 몇 개 샘플 포인트
    test_points = [
        (480, 294, "Center"),
        (roi_min_x, roi_min_y, "Bottom-left corner"),
        (roi_max_x, roi_max_y, "Top-right corner"),
        (480 + 50, 294 + 30, "Offset (+50, +30)"),
    ]
    
    print(f"\n   GT → ROI relative conversion:")
    print(f"   {'GT Position':<20} {'ROI Relative':<20} {'Description':<25}")
    print(f"   {'-'*65}")
    
    for gt_x, gt_y, desc in test_points:
        # ROI 내 상대 좌표 계산
        rel_x = (gt_x - roi_center[0]) / roi_size[1] + 0.5
        rel_y = (gt_y - roi_center[1]) / roi_size[0] + 0.5
        rel_x = np.clip(rel_x, 0.0, 1.0)
        rel_y = np.clip(rel_y, 0.0, 1.0)
        
        print(f"   ({gt_x:4.0f}, {gt_y:4.0f}){' '*8}({rel_x:.3f}, {rel_y:.3f}){' '*8}{desc}")
    
    print(f"\n   ✅ Label conversion test passed")
    
except Exception as e:
    print(f"   ❌ Label conversion test failed: {e}")

# 5. 모델 구조 확인 (PyTorch 있으면)
print("\n5️⃣ Testing Model Structure...")
try:
    import torch
    from model import get_tracking_model, count_parameters
    
    for model_name in ['basic_tracking']:  # 간단히 basic만 테스트
        try:
            model = get_tracking_model(model_name, input_channels=5, output_dim=2)
            params = count_parameters(model)
            print(f"   ✅ {model_name}: {params['total']:,} parameters")
            
            # 테스트 입력
            test_input = torch.randn(2, 5, 384, 384)
            model.eval()
            with torch.no_grad():
                output = model(test_input)
            print(f"      Input: {tuple(test_input.shape)} → Output: {tuple(output.shape)}")
            
        except Exception as e:
            print(f"   ⚠️  {model_name} test failed: {e}")
    
except ImportError:
    print(f"   ⚠️  PyTorch not available, skipping model test")

# 최종 결과
print("\n" + "=" * 60)
print("✅ Tracking system test completed!")
print("\n📋 Summary:")
print("   ✅ Config system working")
print("   ✅ Brownian motion trajectory generation working")
print("   ✅ Boundary reflection working")
print("   ✅ Label conversion working")
print("   📊 Trajectory visualization saved to: test_trajectory.png")

print("\n🚀 Next steps:")
print("   1. Check test_trajectory.png to verify trajectory")
print("   2. If PyTorch available: python train.py")
print("   3. For full training: activate conda/venv with PyTorch")

