# AICUP 2025 主動脈瓣膜物件偵測專案

## 專案概述
本專案使用 YOLOv11/v12 進行主動脈瓣膜（Aortic Valve）的物件偵測，提供完整的資料處理、訓練、預測和結果優化流程。

## 目錄結構
```
AICUP/
├── README.md                           # 本文件
├── aortic_valve_colab.yaml            # YOLO 資料集配置檔
├── analyze_dataset.py                  # 資料集分析工具
├── train_yolo12s.py                   # YOLOv12-small 訓練腳本
├── train_advanced.py                  # 進階資料增強訓練
├── optimize_prediction.py             # 優化預測（TTA、多尺度）
├── ensemble_predictions.py            # 模型集成預測
├── yolo11n.pt                         # YOLOv11-nano 預訓練權重
├── yolo12n.pt                         # YOLOv12-nano 預訓練權重
├── yolo12s.pt                         # YOLOv12-small 預訓練權重
├── AI_CUP_2025_aortic_valve_object_detection_train.ipynb  # Colab 訓練筆記本
└── train.ipynb                        # 本地訓練筆記本
```

---

## 環境安裝與配置

### 1. 系統需求
- **作業系統**: Windows 10/11, Linux, macOS
- **Python**: 3.8 以上
- **GPU**: NVIDIA GPU (建議 8GB 以上顯存，如 RTX 3060 以上)
- **CUDA**: 11.7 以上 (若使用 GPU)
- **記憶體**: 建議 16GB 以上

### 2. 建立 Python 環境
```bash
# 使用 Conda (推薦)
conda create -n aicup python=3.10
conda activate aicup

# 或使用 venv
python -m venv aicup_env
# Windows
aicup_env\Scripts\activate
# Linux/Mac
source aicup_env/bin/activate
```

### 3. 安裝依賴套件
```bash
# 安裝 PyTorch (GPU 版本，請依據 CUDA 版本調整)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安裝 Ultralytics (YOLO)
pip install ultralytics

# 安裝其他必要套件
pip install opencv-python
pip install numpy
pip install matplotlib
pip install Pillow
pip install pyyaml
```

### 4. 驗證安裝
```python
import torch
import ultralytics

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"Ultralytics 版本: {ultralytics.__version__}")
```

---

## 資料準備

### 1. 資料集結構
請將資料集整理為以下結構：
```
datasets/
├── train/
│   ├── images/
│   │   ├── patient001_frame001.jpg
│   │   ├── patient001_frame002.jpg
│   │   └── ...
│   └── labels/
│       ├── patient001_frame001.txt
│       ├── patient001_frame002.txt
│       └── ...
└── val/
    ├── images/
    └── labels/
```

### 2. 標籤格式
YOLO 格式標籤（.txt），每行一個物件：
```
<class_id> <x_center> <y_center> <width> <height>
```
- 所有座標為相對值 (0-1)
- class_id: 0 表示 aortic_valve

### 3. 配置 YAML 檔案
修改 `aortic_valve_colab.yaml`：
```yaml
path: ./datasets
train: train/images
val: val/images

nc: 1
names:
  0: aortic_valve
```

---

## 使用指南

### 模組 1: 資料集分析 (`analyze_dataset.py`)

#### 功能
- 統計影像總數
- 計算正負樣本比例
- 分析框的尺寸分佈
- 視覺化資料分佈

#### 輸入
- `IMAGE_DIR`: 影像資料夾路徑
- `LABEL_DIR`: 標籤資料夾路徑

#### 輸出
- 控制台輸出統計資訊
- 視覺化圖表（框尺寸分佈、類別分佈等）

#### 使用方法
```bash
# 修改腳本中的路徑
IMAGE_DIR = "42_training_image/training_image"
LABEL_DIR = "42_training_label/training_label"

# 執行分析
python analyze_dataset.py
```

#### 範例輸出
```
總影像數量: 1200
正樣本數量: 950
負樣本數量: 250
正樣本比例: 79.17%
...
```

---

### 模組 2: 基礎訓練 (`train_yolo12s.py`)

#### 功能
使用 YOLOv12-small 進行完整訓練，適合追求高精度

#### 輸入
- 資料集 YAML: `aortic_valve_colab.yaml`
- 預訓練權重: `yolo12s.pt`

#### 輸出
- 訓練權重: `runs/detect/train/weights/best.pt`, `last.pt`
- 訓練日誌: `runs/detect/train/results.csv`
- 視覺化結果: 混淆矩陣、PR 曲線、訓練曲線

#### 訓練參數說明
```python
epochs=100           # 訓練輪數
batch=4             # 批次大小（依 GPU 記憶體調整）
imgsz=640           # 輸入影像尺寸
optimizer='AdamW'   # 優化器
lr0=0.001          # 初始學習率
```

#### 使用方法
```bash
python train_yolo12s.py
```

#### 預期效果
- 訓練時間: 2-3 小時 (RTX 3090 Ti)
- mAP50: 95-97%
- mAP50-95: 85-90%

---

### 模組 3: 進階訓練 (`train_advanced.py`)

#### 功能
使用進階資料增強技術，針對醫學影像特性優化

#### 特點
- 保守的 HSV 調整（保護醫學影像色彩）
- 適度的幾何變換
- 混合增強（Mixup, Mosaic）
- 持續學習策略

#### 輸入
- 前一階段最佳模型: `runs/detect/train/weights/best.pt`（來自基礎訓練）
- 資料集 YAML: `aortic_valve_colab.yaml`

#### 輸出
- 增強後的模型權重: `runs/detect/train_advanced/weights/best.pt`, `last.pt`
- 訓練日誌和視覺化

#### 使用方法
```bash
# 執行前請先修改腳本中的模型路徑
# 將 line 20 的路徑改為: './runs/detect/train/weights/best.pt'
python train_advanced.py
```

#### 增強參數
```python
hsv_h=0.01    # 色調調整（降低以保護醫學影像）
hsv_s=0.3     # 飽和度
hsv_v=0.2     # 亮度
degrees=10    # 旋轉角度
translate=0.1 # 平移
scale=0.2     # 縮放
flipud=0.0    # 上下翻轉（醫學影像不建議）
fliplr=0.3    # 左右翻轉
mosaic=0.5    # Mosaic 增強
mixup=0.1     # Mixup 增強
```

---

### 模組 4: 優化預測 (`optimize_prediction.py`)

#### 功能
提供多種預測增強技術以提升精度

#### 4.1 TTA (Test Time Augmentation) 預測
**輸入**:
- 模型權重: 使用最佳訓練模型（例如 `runs/detect/train/weights/best.pt` 或 `runs/detect/train_advanced/weights/best.pt`）
- 測試影像資料夾

**輸出**:
- 預測結果: `runs/detect/predict_tta_optimized/labels/`
- 每個 .txt 檔案包含: `class x_center y_center width height confidence`

**使用方法**:
```python
# 執行前請修改腳本中的 best_model 路徑（line 16）
# 指向你的最佳模型，例如: './runs/detect/train/weights/best.pt'
from optimize_prediction import predict_with_tta
predict_with_tta()
```

**參數說明**:
```python
imgsz=640       # 影像尺寸
conf=0.25       # 信心度閾值
iou=0.45        # NMS IoU 閾值
augment=True    # 開啟 TTA
```

#### 4.2 多尺度預測
**輸入**:
- 模型權重
- 測試影像

**輸出**:
- 三個不同尺度的預測結果

**使用方法**:
```python
from optimize_prediction import predict_multi_scale
predict_multi_scale()
```

**尺度設定**:
- 640x640: 標準尺寸
- 800x800: 提升小物件檢測
- 960x960: 最大化細節

#### 4.3 置信度優化
**輸入**:
- 原始預測結果資料夾

**輸出**:
- 優化後的預測結果

**使用方法**:
```python
from optimize_prediction import optimize_confidence
optimize_confidence()
```

---

### 模組 5: 集成預測 (`ensemble_predictions.py`)

#### 功能
融合多個模型的預測結果以提升精度和穩定性

#### 5.1 簡單集成（NMS）
**輸入**:
```python
prediction_dirs = [
    'runs/detect/predict1/labels',
    'runs/detect/predict2/labels',
    'runs/detect/predict3/labels'
]
```

**輸出**:
- 集成結果: `runs/detect/ensemble_nms/labels/`

**使用方法**:
```bash
python ensemble_predictions.py
```

#### 5.2 加權集成
**權重設定**:
```python
model_weights = [0.4, 0.35, 0.25]  # 依模型效能分配
```

**融合參數**:
```python
iou_threshold=0.5   # IoU 閾值，超過此值視為同一物件
conf_threshold=0.3  # 最低信心度閾值
```

#### 集成策略
1. **多模型融合**: 結合 YOLOv11n, YOLOv12n, YOLOv12s
2. **多尺度融合**: 融合不同輸入尺寸的預測
3. **TTA 融合**: 融合不同增強的預測

---

## 完整訓練流程

### 步驟 1: 環境準備
```bash
# 啟動環境
conda activate aicup

# 驗證 GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### 步驟 2: 資料分析（建議執行）
```bash
# 分析資料集分佈、統計資訊
python analyze_dataset.py
```
**目的**: 了解資料集特性，包括正負樣本比例、框尺寸分佈等，有助於後續參數調整。

---

### 步驟 3: 基礎訓練
```bash
# 使用 YOLOv12-small 訓練
python train_yolo12s.py
```
**產出**: `runs/detect/train/weights/best.pt`

### 步驟 4: 評估基礎模型
```bash
# 驗證訓練結果
yolo val model=runs/detect/train/weights/best.pt data=aortic_valve_colab.yaml
```
**檢查**: 查看 mAP50 是否達到 95% 以上。如果已滿足需求，可直接跳到步驟 7 進行預測。

---

### 步驟 5: 進階訓練（可選，追求更高精度）
```bash
# 基於最佳模型進行進階增強訓練
python train_advanced.py
```
**前提**: 需先修改 `train_advanced.py` 中的模型路徑為 `runs/detect/train/weights/best.pt`
**產出**: `runs/detect/train_advanced/weights/best.pt`

### 步驟 6: 評估進階模型（若執行步驟 5）
```bash
# 驗證進階訓練結果
yolo val model=runs/detect/train_advanced/weights/best.pt data=aortic_valve_colab.yaml
```
**比較**: 對比步驟 4 的結果，選擇表現最佳的模型。

---

### 步驟 7: 選擇最佳模型並預測
```bash
# 方案 A: 使用 TTA 預測（推薦）
# 需修改 optimize_prediction.py 中的 best_model 路徑
python -c "from optimize_prediction import predict_with_tta; predict_with_tta()"

# 方案 B: 多尺度預測
python -c "from optimize_prediction import predict_multi_scale; predict_multi_scale()"

# 方案 C: 直接使用 YOLO 命令預測
yolo predict model=runs/detect/train/weights/best.pt source=測試影像路徑 conf=0.25
```
**注意**: 使用 Python 腳本前，請先修改腳本中的模型路徑指向你選擇的最佳模型。

### 步驟 8: 結果集成（可選，進一步提升精度）
```bash
# 融合多個模型或多尺度的預測結果
python ensemble_predictions.py
```
**前提**: 需要有多個預測結果資料夾可供融合。

---

## 除錯指南

### 常見問題與解決方案

#### 1. CUDA Out of Memory
**錯誤**: `RuntimeError: CUDA out of memory`

**解決方案**:
```python
# 減小批次大小
batch=2  # 從 4 改為 2

# 或減小影像尺寸
imgsz=512  # 從 640 改為 512
```

#### 2. 找不到資料集
**錯誤**: `Dataset not found`

**解決方案**:
```yaml
# 檢查 aortic_valve_colab.yaml 路徑
path: ./datasets  # 使用絕對路徑或相對路徑
train: train/images
val: val/images
```

#### 3. 模型無法載入
**錯誤**: `Model file not found`

**解決方案**:
```bash
# 下載預訓練權重
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo12s.pt
```

#### 4. 訓練效果不佳

**檢查清單**:
- [ ] 資料集標註是否正確
- [ ] 資料增強是否過度
- [ ] 學習率是否合適
- [ ] 訓練輪數是否足夠

**優化建議**:
```python
# 調整學習率
lr0=0.0005  # 降低學習率

# 增加訓練輪數
epochs=150

# 調整損失權重
box=7.5
cls=0.5
```

#### 5. 預測信心度過低

**解決方案**:
```python
# 降低信心度閾值
conf=0.2  # 從 0.25 降至 0.2

# 使用 TTA
augment=True

# 多尺度預測
predict_multi_scale()
```

---

## 效能指標

### 評估指標說明

#### mAP (mean Average Precision)
- **mAP50**: IoU=0.5 時的平均精度
- **mAP50-95**: IoU 從 0.5 到 0.95 的平均精度
- 目標: mAP50 > 95%

#### Precision & Recall
- **Precision**: 預測為正的樣本中實際為正的比例
- **Recall**: 實際為正的樣本中被正確預測的比例
- 目標: Precision > 90%, Recall > 90%

#### F1-Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### 模型效能比較

| 模型 | mAP50 | mAP50-95 | 推論速度 (ms) | 模型大小 |
|------|-------|----------|---------------|----------|
| YOLOv11n | 92-94% | 80-83% | 8-10 | 5.3 MB |
| YOLOv12n | 93-95% | 82-85% | 9-11 | 5.5 MB |
| YOLOv12s | 95-97% | 85-90% | 12-15 | 21.5 MB |

---

## 重現結果

### 完整重現步驟

```bash
# 1. 環境設定
conda create -n aicup python=3.10
conda activate aicup
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy matplotlib pyyaml

# 2. 準備資料集（假設已下載）
# 確保資料集結構符合 YOLO 格式: datasets/train, datasets/val

# 3. 基礎訓練
python train_yolo12s.py

# 4. 評估模型
yolo val model=runs/detect/train/weights/best.pt data=aortic_valve_colab.yaml

# 5. （可選）進階訓練 - 需先修改腳本路徑
# 編輯 train_advanced.py，將 line 20 改為: './runs/detect/train/weights/best.pt'
python train_advanced.py

# 6. 預測 - 需先修改腳本路徑
# 編輯 optimize_prediction.py，將 line 16 改為你的最佳模型路徑
python -c "from optimize_prediction import predict_with_tta; predict_with_tta()"

# 7. （可選）集成多模型預測
python ensemble_predictions.py
```

### 隨機種子設定
為確保可重現性，在訓練腳本中設定：
```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

---

## 重要檔案說明

### 輸入檔案
1. **aortic_valve_colab.yaml**: 資料集配置
2. **yolo12s.pt**: 預訓練權重
3. **datasets/**: 訓練和驗證資料

### 輸出檔案
1. **runs/detect/train/weights/best.pt**: 最佳模型權重
2. **runs/detect/train/results.csv**: 訓練日誌
3. **runs/detect/predict_tta_optimized/labels/**: 預測結果
4. **runs/detect/ensemble_nms/labels/**: 集成結果

### 日誌檔案
- **results.csv**: 每個 epoch 的指標
- **confusion_matrix.png**: 混淆矩陣
- **PR_curve.png**: Precision-Recall 曲線
- **F1_curve.png**: F1 分數曲線

---

## 進階技巧

### 1. 超參數調優
使用 Ultralytics 內建的超參數演化：
```bash
yolo train data=aortic_valve_colab.yaml model=yolo12s.pt epochs=100 evolve=300
```

### 2. 模型剪枝
減少模型大小並加速推論：
```python
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
model.export(format='onnx', simplify=True)
```

### 3. 量化加速
轉換為 INT8 量化模型：
```python
model.export(format='tflite', int8=True)
```

### 4. 自定義資料增強
```python
# 在訓練腳本中添加
from albumentations import *

transform = Compose([
    RandomBrightnessContrast(p=0.5),
    GaussNoise(p=0.3),
    Blur(p=0.3)
])
```

---

## 授權與引用

### 依賴項目
- [Ultralytics YOLOv8/11/12](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)

### 引用
如果本專案對您的研究有幫助，請引用：
```
@misc{aicup2025_aortic_valve,
  title={AICUP 2025 Aortic Valve Object Detection},
  author={Your Name},
  year={2025}
}
```

---

## 聯絡資訊

- **專案負責人**: [您的姓名]
- **Email**: [您的信箱]
- **GitHub**: [專案連結]

---

## 更新日誌

### 2026-01-08
- ✅ 初始版本發布
- ✅ 完成基礎訓練流程
- ✅ 添加 TTA 和多尺度預測
- ✅ 實現模型集成功能
- ✅ 完成文檔撰寫

---

## 附錄

### A. 硬體需求建議

| 配置等級 | GPU | 記憶體 | 訓練時間 (100 epochs) |
|---------|-----|--------|---------------------|
| 最低 | GTX 1660 6GB | 8GB | ~6 小時 |
| 建議 | RTX 3060 12GB | 16GB | ~3 小時 |
| 推薦 | RTX 3090/4090 | 32GB | ~1.5 小時 |

### B. 資料增強範例

```python
# 醫學影像友好的增強設定
augmentation_config = {
    'hsv_h': 0.01,      # 最小色調變化
    'hsv_s': 0.3,       # 適度飽和度
    'hsv_v': 0.2,       # 適度亮度
    'degrees': 10,      # 小角度旋轉
    'translate': 0.1,   # 小範圍平移
    'scale': 0.2,       # 適度縮放
    'flipud': 0.0,      # 不上下翻轉
    'fliplr': 0.3,      # 可左右翻轉
    'mosaic': 0.5,      # Mosaic 增強
    'mixup': 0.1        # Mixup 增強
}
```

### C. 檢查清單

訓練前檢查：
- [ ] GPU 驅動已安裝
- [ ] CUDA 環境已配置
- [ ] Python 套件已安裝
- [ ] 資料集路徑正確
- [ ] YAML 配置正確
- [ ] 預訓練權重已下載

預測前檢查：
- [ ] 模型權重存在
- [ ] 測試影像路徑正確
- [ ] 輸出資料夾可寫入

---

**祝您訓練順利！如有問題請參考除錯指南或聯繫專案維護者。**
