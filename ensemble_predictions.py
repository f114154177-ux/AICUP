#!/usr/bin/env python3
"""
集成多個預測結果以提升最終精度
使用 Weighted Boxes Fusion (WBF) 或 Non-Maximum Weighted (NMW)
"""
import numpy as np
from pathlib import Path
from collections import defaultdict
import shutil

def read_yolo_labels(label_path):
    """讀取 YOLO 格式標籤"""
    boxes = []
    if Path(label_path).exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # class, x_center, y_center, width, height, [confidence]
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    conf = float(parts[5]) if len(parts) > 5 else 1.0
                    boxes.append([cls, x, y, w, h, conf])
    return boxes


def write_yolo_labels(label_path, boxes):
    """寫入 YOLO 格式標籤"""
    with open(label_path, 'w') as f:
        for box in boxes:
            cls, x, y, w, h, conf = box
            f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")


def iou(box1, box2):
    """計算兩個框的 IoU"""
    # box format: [cls, x_center, y_center, width, height, conf]
    x1_min = box1[1] - box1[3] / 2
    y1_min = box1[2] - box1[4] / 2
    x1_max = box1[1] + box1[3] / 2
    y1_max = box1[2] + box1[4] / 2
    
    x2_min = box2[1] - box2[3] / 2
    y2_min = box2[2] - box2[4] / 2
    x2_max = box2[1] + box2[3] / 2
    y2_max = box2[2] + box2[4] / 2
    
    # 計算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    # 計算聯集
    box1_area = box1[3] * box1[4]
    box2_area = box2[3] * box2[4]
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def nms_fusion(all_boxes, iou_threshold=0.5):
    """使用 NMS 融合多個預測結果"""
    if not all_boxes:
        return []
    
    # 按信心分數排序
    sorted_boxes = sorted(all_boxes, key=lambda x: x[5], reverse=True)
    
    keep_boxes = []
    while sorted_boxes:
        # 取出信心最高的框
        best_box = sorted_boxes.pop(0)
        keep_boxes.append(best_box)
        
        # 移除與 best_box 重疊度高的框
        sorted_boxes = [
            box for box in sorted_boxes
            if iou(best_box, box) < iou_threshold
        ]
    
    return keep_boxes


def weighted_boxes_fusion(all_boxes, iou_threshold=0.5):
    """使用加權平均融合重疊的框"""
    if not all_boxes:
        return []
    
    # 按信心分數排序
    sorted_boxes = sorted(all_boxes, key=lambda x: x[5], reverse=True)
    
    fused_boxes = []
    used = [False] * len(sorted_boxes)
    
    for i, box1 in enumerate(sorted_boxes):
        if used[i]:
            continue
        
        # 找出所有與 box1 重疊的框
        cluster = [box1]
        used[i] = True
        
        for j, box2 in enumerate(sorted_boxes[i+1:], start=i+1):
            if not used[j] and iou(box1, box2) >= iou_threshold:
                cluster.append(box2)
                used[j] = True
        
        # 加權平均融合
        if len(cluster) == 1:
            fused_boxes.append(cluster[0])
        else:
            # 使用信心分數作為權重
            weights = np.array([box[5] for box in cluster])
            weights = weights / weights.sum()
            
            cls = cluster[0][0]  # 類別取第一個
            x = sum(box[1] * w for box, w in zip(cluster, weights))
            y = sum(box[2] * w for box, w in zip(cluster, weights))
            w = sum(box[3] * w for box, w in zip(cluster, weights))
            h = sum(box[4] * w for box, w in zip(cluster, weights))
            conf = max(box[5] for box in cluster)  # 信心取最大值
            
            fused_boxes.append([cls, x, y, w, h, conf])
    
    return fused_boxes


def ensemble_predictions(predict_dirs, output_dir, method='wbf', iou_threshold=0.5, weights=None):
    """
    集成多個預測結果
    
    Args:
        predict_dirs: 預測結果目錄列表
        output_dir: 輸出目錄
        method: 'nms' 或 'wbf' (weighted boxes fusion)
        iou_threshold: IoU 閾值
        weights: 各模型權重（僅用於 wbf）
    """
    print(f"\n{'='*60}")
    print(f"🔄 集成預測結果 - 方法: {method.upper()}")
    print(f"{'='*60}\n")
    
    # 確保輸出目錄存在
    output_path = Path(output_dir) / 'labels'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 收集所有預測目錄中的標籤文件
    all_label_files = defaultdict(list)
    
    for pred_dir in predict_dirs:
        label_dir = Path(pred_dir) / 'labels'
        if not label_dir.exists():
            print(f"⚠️  警告: {label_dir} 不存在，跳過")
            continue
        
        print(f"📂 處理目錄: {pred_dir}")
        for label_file in label_dir.glob('*.txt'):
            all_label_files[label_file.name].append(label_file)
    
    # 為每個圖像融合預測
    total_files = len(all_label_files)
    print(f"\n📊 總共 {total_files} 個圖像需要處理")
    
    processed = 0
    for filename, label_paths in all_label_files.items():
        # 讀取所有預測
        all_boxes = []
        for label_path in label_paths:
            boxes = read_yolo_labels(label_path)
            all_boxes.extend(boxes)
        
        # 融合預測
        if method == 'wbf':
            fused_boxes = weighted_boxes_fusion(all_boxes, iou_threshold)
        else:  # nms
            fused_boxes = nms_fusion(all_boxes, iou_threshold)
        
        # 寫入結果
        output_file = output_path / filename
        write_yolo_labels(output_file, fused_boxes)
        
        processed += 1
        if processed % 100 == 0:
            print(f"  ✓ 已處理 {processed}/{total_files} 個文件")
    
    print(f"\n✅ 完成！結果儲存於: {output_path}")
    print(f"{'='*60}\n")


def main():
    """主函數"""
    print("\n🎯 預測結果集成工具\n")
    
    # 定義要集成的預測目錄
    predict_dirs = [
        './runs/detect/predict_tta_optimized',
        './runs/detect/predict_tta',
        './runs/detect/predict_tta2',
        './runs/detect/predict3',
    ]
    
    # 過濾存在的目錄
    existing_dirs = [d for d in predict_dirs if Path(d).exists()]
    
    if not existing_dirs:
        print("❌ 錯誤: 找不到任何預測結果目錄")
        print("請先執行 optimize_prediction.py 生成預測結果")
        return
    
    print("找到以下預測結果:")
    for i, d in enumerate(existing_dirs, 1):
        print(f"  {i}. {d}")
    
    print(f"\n將使用 {len(existing_dirs)} 個預測結果進行集成")
    
    # 執行集成
    output_dir = './runs/detect/ensemble_final'
    
    print("\n選擇集成方法:")
    print("1. WBF (Weighted Boxes Fusion) - 推薦")
    print("2. NMS (Non-Maximum Suppression)")
    
    choice = input("\n請選擇 (1-2，直接 Enter 使用 WBF): ").strip()
    method = 'nms' if choice == '2' else 'wbf'
    
    ensemble_predictions(
        predict_dirs=existing_dirs,
        output_dir=output_dir,
        method=method,
        iou_threshold=0.5
    )
    
    print("🎉 集成完成！")
    print(f"📁 結果位置: {output_dir}/labels")


if __name__ == '__main__':
    main()
