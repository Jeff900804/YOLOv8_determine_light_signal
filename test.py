from ultralytics import YOLO

# 載入你訓練好的模型
model = YOLO("runs/train/light_train/weights/best.pt")

# 要測試的圖片（你可以換成其他圖）
img_path = "test_image/20250408 (1).png"

# 推論圖片，不要用 show（避免 GUI 問題）
results = model(img_path)

# 將結果儲存成圖片（會自動畫框＋標籤）
results[0].save(filename=img_path[0:-4]+"_result.png")
