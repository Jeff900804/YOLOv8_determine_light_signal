
from ultralytics import YOLO

# 載入 YOLOv8 預訓練模型（輕量版 yolov8n）
model = YOLO('yolov8n.pt')

# 訓練模型
model.train(
    data='/data/train/light-dataset/data.yaml',  # 資料集路徑
    epochs=50,                       # 訓練輪數
    imgsz=640,                       # 圖片尺寸
    batch=4,                         # 批次大小
    name='light_train',             # 訓練結果資料夾名稱
    project='runs/train'            # 儲存位置
)

# 驗證（也可以單獨跑這段）
metrics = model.val()
print(metrics)
