#!/usr/bin/env python3
"""
優化預測腳本 - 使用最佳模型和多種增強技術
"""
from ultralytics import YOLO
import torch
from pathlib import Path

def predict_with_tta():
    """使用 TTA 進行預測"""
    print("=" * 60)
    print("🎯 使用 TTA 增強預測")
    print("=" * 60)
    
    # 載入最佳模型
    best_model = './runs/detect/train_unfreeze4/weights/best.pt'
    model = YOLO(best_model)
    
    # TTA 預測設定
    results = model.predict(
        source='./testing_image/42_testing_image/testing_image',
        imgsz=640,
        conf=0.25,  # 信心閾值
        iou=0.45,   # NMS IoU 閾值
        augment=True,  # 開啟 TTA
        device=0,
        save_txt=True,
        save_conf=True,
        project='runs/detect',
        name='predict_tta_optimized'
    )
    print(f"✅ TTA 預測完成！結果儲存於: runs/detect/predict_tta_optimized")


def predict_multi_scale():
    """多尺度預測並融合"""
    print("\n" + "=" * 60)
    print("🎯 多尺度預測")
    print("=" * 60)
    
    best_model = './runs/detect/train_unfreeze4/weights/best.pt'
    model = YOLO(best_model)
    
    scales = [640, 800, 960]  # 不同圖像尺寸
    
    for scale in scales:
        print(f"\n📏 預測尺寸: {scale}x{scale}")
        results = model.predict(
            source='./testing_image/42_testing_image/testing_image',
            imgsz=scale,
            conf=0.20,  # 較低閾值以獲取更多候選
            iou=0.45,
            augment=True,
            device=0,
            save_txt=True,
            save_conf=True,
            project='runs/detect',
            name=f'predict_scale_{scale}'
        )
        print(f"✅ 尺寸 {scale} 預測完成！")


def predict_ensemble():
    """集成多個模型的預測"""
    print("\n" + "=" * 60)
    print("🎯 模型集成預測")
    print("=" * 60)
    
    # 使用多個訓練 checkpoint
    models = [
        './runs/detect/train_unfreeze4/weights/best.pt',
        './runs/detect/train_freeze4/weights/best.pt',
    ]
    
    for idx, model_path in enumerate(models):
        if Path(model_path).exists():
            print(f"\n🤖 模型 {idx+1}: {model_path}")
            model = YOLO(model_path)
            results = model.predict(
                source='./testing_image/42_testing_image/testing_image',
                imgsz=640,
                conf=0.20,
                iou=0.45,
                augment=True,
                device=0,
                save_txt=True,
                save_conf=True,
                project='runs/detect',
                name=f'predict_ensemble_{idx+1}'
            )
            print(f"✅ 模型 {idx+1} 預測完成！")


def predict_optimized_threshold():
    """優化信心閾值"""
    print("\n" + "=" * 60)
    print("🎯 閾值優化預測")
    print("=" * 60)
    
    best_model = './runs/detect/train_unfreeze4/weights/best.pt'
    model = YOLO(best_model)
    
    # 嘗試不同的信心閾值
    conf_thresholds = [0.15, 0.20, 0.25, 0.30]
    
    for conf in conf_thresholds:
        print(f"\n📊 信心閾值: {conf}")
        results = model.predict(
            source='./testing_image/42_testing_image/testing_image',
            imgsz=640,
            conf=conf,
            iou=0.45,
            augment=True,
            device=0,
            save_txt=True,
            save_conf=True,
            project='runs/detect',
            name=f'predict_conf_{conf}'
        )
        print(f"✅ 閾值 {conf} 預測完成！")


if __name__ == '__main__':
    print("\n🚀 開始優化預測流程\n")
    
    # 檢查 CUDA
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 執行各種預測策略
    print("\n選擇預測策略:")
    print("1. TTA 增強預測（推薦）")
    print("2. 多尺度預測")
    print("3. 模型集成")
    print("4. 閾值優化")
    print("5. 全部執行")
    
    choice = input("\n請選擇 (1-5): ").strip()
    
    if choice == '1':
        predict_with_tta()
    elif choice == '2':
        predict_multi_scale()
    elif choice == '3':
        predict_ensemble()
    elif choice == '4':
        predict_optimized_threshold()
    elif choice == '5':
        predict_with_tta()
        predict_multi_scale()
        predict_ensemble()
        predict_optimized_threshold()
    else:
        print("❌ 無效選擇，執行預設 TTA 預測")
        predict_with_tta()
    
    print("\n" + "=" * 60)
    print("🎉 預測流程完成！")
    print("=" * 60)
