#!/usr/bin/env python3
"""
使用 YOLOv12-small 模型進行訓練以提升精度
"""
from ultralytics import YOLO
import torch

def train_yolo12s():
    """使用 YOLOv12-small 進行完整訓練"""
    
    print("=" * 60)
    print("🚀 YOLOv12-small 完整訓練")
    print("=" * 60)
    
    print(f'\n✅ CUDA 可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    
    # 載入 YOLOv12-small 預訓練模型
    model = YOLO('yolo12s.pt')
    
    print('\n📊 開始訓練 YOLOv12-small...')
    print('預期訓練時間: 約 2-3 小時')
    print('預期 mAP50: 95-97%')
    
    # 訓練參數 - 針對小型資料集優化
    results = model.train(
        # 基本設定
        data='./aortic_valve_colab.yaml',
        epochs=100,
        batch=4,  # RTX 3090 Ti 24GB 可以用 4-8
        imgsz=640,
        
        # 優化器設定
        optimizer='AdamW',  # AdamW 通常比 SGD 更穩定
        lr0=0.001,  # 初始學習率
        lrf=0.01,   # 最終學習率 = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        
        # 學習率預熱
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # 損失函數權重
        box=7.5,    # box loss 權重
        cls=0.5,    # classification loss 權重
        dfl=1.5,    # DFL loss 權重
        
        # 資料增強 - HSV
        hsv_h=0.015,  # 色調抖動
        hsv_s=0.7,    # 飽和度抖動
        hsv_v=0.4,    # 明度抖動
        
        # 資料增強 - 幾何變換
        degrees=10.0,      # 旋轉角度
        translate=0.1,     # 平移
        scale=0.5,         # 縮放
        shear=0.0,         # 剪切
        perspective=0.0,   # 透視變換
        flipud=0.5,        # 上下翻轉
        fliplr=0.5,        # 左右翻轉
        
        # 資料增強 - 混合
        mosaic=1.0,      # Mosaic 增強
        mixup=0.0,       # Mixup 增強
        copy_paste=0.0,  # Copy-paste 增強
        
        # 硬體設定
        device=0,
        workers=8,
        amp=True,  # 混合精度訓練
        
        # 早停與儲存
        patience=20,  # 早停耐心值
        save=True,
        save_period=10,  # 每 10 epochs 儲存一次
        
        # 輸出設定
        project='runs/detect',
        name='train_yolo12s_best',
        exist_ok=True,
        
        # 驗證設定
        val=True,
        plots=True,
        verbose=True
    )
    
    print('\n' + "=" * 60)
    print('✅ 訓練完成！')
    print("=" * 60)
    print(f'\n最佳模型位置: runs/detect/train_yolo12s_best/weights/best.pt')
    
    return results


if __name__ == '__main__':
    print("\n🎯 使用 YOLOv12-small 提升模型精度")
    print("相比 nano 版本，small 模型具有:")
    print("  • 更多參數 (11M vs 3M)")
    print("  • 更強的特徵提取能力")
    print("  • 預期 mAP 提升 2-4%")
    print("\n按 Enter 開始訓練，或 Ctrl+C 取消...")
    input()
    
    train_yolo12s()
