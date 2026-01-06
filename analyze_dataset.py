import os
import glob
from collections import Counter, defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 路徑依你實際情況修改 ===
IMAGE_DIR = "42_training_image/training_image"   # 影像資料夾
LABEL_DIR = "42_training_label/training_label"    # YOLO .txt 標註資料夾
IMG_EXTS = [".jpg", ".png", ".jpeg"]        # 可能的影像副檔名

# 如果只把某個 class 視為「正樣本」，在這裡設定，例如 0
POSITIVE_CLASS_IDS = {0}    # 如果全部類別都算正樣本，就改成 set(range(100)) 或直接不判斷

def list_images(image_dir):
    """遍歷所有 patient 資料夾下的影像"""
    paths = []
    patient_dirs = sorted([d for d in os.listdir(image_dir) 
                          if os.path.isdir(os.path.join(image_dir, d)) and d.startswith('patient')])
    for patient_dir in patient_dirs:
        patient_path = os.path.join(image_dir, patient_dir)
        for ext in IMG_EXTS:
            paths.extend(glob.glob(os.path.join(patient_path, f"*{ext}")))
    return sorted(paths)

def read_yolo_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            # 處理 class 可能是 0.000 這種浮點格式的情況
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])
            boxes.append((cls, xc, yc, w, h))
    return boxes

def main():
    image_paths = list_images(IMAGE_DIR)
    total_images = len(image_paths)
    print(f"總影像數量: {total_images}")

    num_positive_imgs = 0
    num_negative_imgs = 0
    all_widths = []
    all_heights = []
    labels_per_image = []
    pos_centers_x = []
    pos_centers_y = []

    class_counter = Counter()

    for img_path in image_paths:
        # 從影像路徑取得 patient 資料夾和檔名
        # 例如: 42_training_image/training_image/patient0001/patient0001_0201.png
        patient_dir = os.path.basename(os.path.dirname(img_path))
        name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABEL_DIR, patient_dir, name + ".txt")
        boxes = read_yolo_labels(label_path)

        # 計算這張圖的總 bbox 數
        labels_per_image.append(len(boxes))

        # 判斷正負樣本（只看指定的 positive class）
        has_positive = any(b[0] in POSITIVE_CLASS_IDS for b in boxes)
        if has_positive:
            num_positive_imgs += 1
        else:
            num_negative_imgs += 1

        # bbox 尺寸與位置統計（以正規化座標為例）
        for cls, xc, yc, w, h in boxes:
            class_counter[cls] += 1
            all_widths.append(w)
            all_heights.append(h)
            if cls in POSITIVE_CLASS_IDS:
                pos_centers_x.append(xc)
                pos_centers_y.append(yc)

    print(f"正樣本圖片數: {num_positive_imgs}")
    print(f"負樣本圖片數: {num_negative_imgs}")
    print(f"所有 bbox 數量: {len(all_widths)}")
    print("各類別 bbox 數量:", class_counter)

    if all_widths:
        print("bbox 寬度 (正規化) - min/avg/max:",
              np.min(all_widths), np.mean(all_widths), np.max(all_widths))
        print("bbox 高度 (正規化) - min/avg/max:",
              np.min(all_heights), np.mean(all_heights), np.max(all_heights))

    # === 繪圖（存檔用於 PPT） ===
    os.makedirs("stats_figs", exist_ok=True)

    # 1. 正負樣本長條圖
    plt.figure()
    plt.bar(["Positive", "Negative"], [num_positive_imgs, num_negative_imgs], color=["#4caf50", "#f44336"])
    plt.title("Positive vs Negative Images")
    plt.ylabel("Number of images")
    plt.savefig("stats_figs/pos_neg_images.png", dpi=150)
    plt.close()

    # 2. bbox 寬度分布
    if all_widths:
        plt.figure()
        plt.hist(all_widths, bins=20, color="#2196f3")
        plt.title("Distribution of bbox width (normalized)")
        plt.xlabel("width (0~1)")
        plt.ylabel("count")
        plt.savefig("stats_figs/bbox_width_hist.png", dpi=150)
        plt.close()

        # 3. bbox 高度分布
        plt.figure()
        plt.hist(all_heights, bins=20, color="#ff9800")
        plt.title("Distribution of bbox height (normalized)")
        plt.xlabel("height (0~1)")
        plt.ylabel("count")
        plt.savefig("stats_figs/bbox_height_hist.png", dpi=150)
        plt.close()

    # 4. 每張圖標註數量分布
    plt.figure()
    plt.hist(labels_per_image, bins=range(0, max(labels_per_image) + 2), align="left", rwidth=0.8)
    plt.title("Number of labels per image")
    plt.xlabel("labels per image")
    plt.ylabel("number of images")
    plt.savefig("stats_figs/labels_per_image_hist.png", dpi=150)
    plt.close()

    # 5. 正樣本位置 heatmap
    if pos_centers_x:
        plt.figure()
        plt.hist2d(pos_centers_x, pos_centers_y, bins=20, range=[[0,1],[0,1]], cmap="hot")
        plt.colorbar(label="count")
        plt.title("Heatmap of positive bbox centers (normalized)")
        plt.xlabel("x center (0~1)")
        plt.ylabel("y center (0~1)")
        plt.gca().invert_yaxis()  # 方便對應影像座標
        plt.savefig("stats_figs/positive_center_heatmap.png", dpi=150)
        plt.close()

    print("圖檔已輸出到 stats_figs/ 資料夾，可直接放進 PPT。")

if __name__ == "__main__":
    main()