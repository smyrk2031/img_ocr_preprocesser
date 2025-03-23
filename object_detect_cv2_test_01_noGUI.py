import cv2
import numpy as np
import matplotlib.pyplot as plt

# ファイルを読み込み グレースケール化
image_path = r"D:\winpythons\WPy64-31180_rinna\venv\app_rakurakuOCR\src\rakurakuOCR_functions\dataset\table_a_01.png"
img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread(image_path)  # カラー画像として読み込み

# 画像が正しく読み込めたかチェック
if img_gray is None:
    print("エラー: 画像ファイルが読み込めません")
    exit()

# しきい値指定によるフィルタリング
_, threshold = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY)

# 輪郭を抽出
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

font = cv2.FONT_HERSHEY_SIMPLEX

# 10色のカラーパレットを作成
colors = [
    (255, 0, 0),    # 青
    (0, 255, 0),    # 緑
    (0, 0, 255),    # 赤
    (255, 255, 0),  # シアン
    (255, 0, 255),  # マゼンタ
    (0, 255, 255),  # イエロー
    (128, 0, 0),    # ダークブルー
    (0, 128, 0),    # ダークグリーン
    (0, 0, 128),    # ダークレッド
    (128, 128, 0)   # オリーブ
]

# 図形の数の変数
triangle = 0
rectangle = 0
pentagon = 0
oval = 0
circle = 0

# 図形の設定
min_area = 500  # 最小面積（ピクセル数）を設定。この値は必要に応じて調整してください

for i, cnt in enumerate(contours):
    # 輪郭の面積を計算
    area = cv2.contourArea(cnt)
    
    # 最小面積より小さい図形はスキップ
    if area < min_area:
        continue
        
    color = colors[i % len(colors)]  # 色を循環させる
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    cv2.drawContours(img_color, [approx], 0, color, 2)
    
    # 図形の左上の座標を取得
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 10  # テキストを図形の少し上に配置

    if len(approx) == 3:
        triangle += 1
        cv2.putText(img_color, f"triangle{triangle}", (x, y), font, 0.4, color, 1)
        
    elif len(approx) == 4:
        rectangle += 1
        cv2.putText(img_color, f"rectangle{rectangle}", (x, y), font, 0.4, color, 1)
                
    elif len(approx) == 5:
        pentagon += 1
        cv2.putText(img_color, f"pentagon{pentagon}", (x, y), font, 0.4, color, 1)
        
    elif 6 < len(approx) < 14:
        oval += 1
        cv2.putText(img_color, f"oval{oval}", (x, y), font, 0.4, color, 1)
        
    else:
        circle += 1
        cv2.putText(img_color, f"circle{circle}", (x, y), font, 0.4, color, 1)
        
# 結果の画像作成
cv2.imwrite('output_shapes.png', img_color)

# 図形の数の結果
print('Number of triangle = ' , triangle)
print('Number of rectangle = ' , rectangle)
print('Number of pentagon = ' , pentagon)
print('Number of circle = ' , circle)
print('Number of oval = ' , oval)


image_path = r"D:\winpythons\WPy64-31180_rinna\venv\app_rakurakuOCR\src\rakurakuOCR_functions\dataset\table_a_01.png"
img = cv2.imread(image_path)
plt.figure(figsize=(10, 10))
img2 = img[:,:,::-1]
plt.xticks([]), plt.yticks([]) 
plt.imshow(img2)