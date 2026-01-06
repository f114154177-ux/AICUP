#!/usr/bin/env python3
"""
進階資料增強訓練 - 針對醫學影像特性優化
"""
from ultralytics import YOLO
import torch

def train_with_advanced_augmentation():
    """使用進階資料增強進行訓練"""
    
    print("=" * 60)
    print("🚀 進階資料增強訓練")
    print("=" * 60)
    
    print(f'\n✅ CUDA 可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    
    # 使用當前最佳模型作為起點
    model = YOLO('./runs/detect/train_unfreeze4/weights/best.pt')
    
    print('\n📊 開始進階增強訓練...')
    print('特點: 針對醫學影像的增強策略')
    
    results = model.train(
        # 基本設定
        data='./aortic_valve_colab.yaml',
        epochs=50,
        batch=4,
        imgsz=640,
        
        # 優化器
        optimizer='AdamW',
        lr0=0.0005,  # 較低學習率用於微調
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # 預熱
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.05,
        
        # 損失權重
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # 醫學影像友好的增強 - 保守的 HSV 調整
        hsv_h=0.01,   # 降低色調變化（醫學影像顏色重要）
        hsv_s=0.5,    # 適度飽和度變化
        hsv_v=0.3,    # 適度明度變化
        
        # 幾何增強 - 考慮超音波特性
        degrees=15.0,      # 增加旋轉（心臟角度變化）
        translate=0.15,    # 增加平移
        scale=0.7,         # 增加縮放範圍
        shear=2.0,         # 輕微剪切
        perspective=0.0001, # 輕微透視變換
        flipud=0.5,        # 上下翻轉（超音波可能的方向）
        fliplr=0.5,        # 左右翻轉
        
        # 進階增強
        mosaic=1.0,        # 保持 Mosaic
        mixup=0.1,         # 添加少量 Mixup
        copy_paste=0.1,    # 添加 Copy-Paste（增加小物體）
        
        # 其他增強
        erasing=0.4,       # 隨機擦除
        crop_fraction=1.0, # 裁剪比例
        
        # 硬體設定
        device=0,
        workers=8,
        amp=True,
        
        # 早停
        patience=15,
        
        # 輸出
        project='runs/detect',
        name='train_advanced_aug',
        exist_ok=True,
        val=True,
        plots=True,
        verbose=True
    )
    
    print('\n' + "=" * 60)
    print('✅ 訓練完成！')
    print("=" * 60)


def train_with_focus_on_hard_examples():
    """專注於困難樣本的訓練"""
    
    print("\n" + "=" * 60)
    print("🎯 困難樣本專注訓練")
    print("=" * 60)
    
    model = YOLO('./runs/detect/train_unfreeze4/weights/best.pt')
    
    results = model.train(
        data='./aortic_valve_colab.yaml',
        epochs=30,
        batch=4,
        imgsz=640,
        
        # 使用 Focal Loss 相關參數
        optimizer='AdamW',
        lr0=0.0003,
        lrf=0.01,
        
        # 增加困難樣本的損失權重
        box=10.0,  # 提高 box loss 權重
        cls=1.0,   # 提高 cls loss 權重
        dfl=2.0,   # 提高 DFL loss 權重
        
        # 其他增強
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.05,
        
        # 硬體設定
        device=0,
        workers=8,
        amp=True,
        patience=15,
        
        project='runs/detect',
        name='train_hard_examples',
        exist_ok=True,
        val=True,
        plots=True
    )
    
    print('\n✅ 困難樣本訓練完成！')


if __name__ == '__main__':
    print("\n🎯 進階訓練策略")
    print("\n選擇訓練方式:")
    print("1. 進階資料增強訓練（推薦）")
    print("2. 困難樣本專注訓練")
    print("3. 兩者都執行")
    
    choice = input("\n請選擇 (1-3): ").strip()
    
    if choice == '1':
        train_with_advanced_augmentation()
    elif choice == '2':
        train_with_focus_on_hard_examples()
    elif choice == '3':
        train_with_advanced_augmentation()
        train_with_focus_on_hard_examples()
    else:
        print("❌ 無效選擇，執行預設訓練")
        train_with_advanced_augmentation()
