import cv2
import numpy as np
import matplotlib.pyplot as plt

# 検出結果の表示に使用する色のリスト
COLORS = [
    (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0),
    (255, 0, 255), (128, 0, 128), (128, 128, 0), (0, 128, 128), (255, 165, 0)
]  # 赤、オレンジ、黄、緑、青、紫、ピンク など

def load_image(path):
    """ 画像を読み込み、グレースケールに変換する関数 """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def correct_skew(image):
    """
    テーブルデータが記載された紙の画像において、テーブルの水平線（または矩形構造）に基づいて画像の傾きを補正する関数

    Parameters
    ----------
    image : numpy.ndarray
        補正対象のカラー画像（BGR形式）または二値化画像

    Returns
    -------
    numpy.ndarray
        傾き補正後の画像。テーブルの水平線が検出できない、または傾きがほぼ水平と判断される場合は元画像を返す。
    """
    # グレースケール化の確認（すでにグレースケールの場合は変換しない）
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Adaptive Thresholdingによる2値化（画像反転して線分を白くする）
    thresh = cv2.adaptiveThreshold(~gray, 255, 
                                  cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 
                                  15, -2)
    
    # 水平線の抽出
    horizontal = thresh.copy()
    cols = horizontal.shape[1]
    # カーネルのサイズは画像サイズに依存（調整可能なパラメータ）
    horizontal_size = cols // 30  
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # モルフォロジー処理で水平線を強調
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)
    
    # Hough変換による水平線の検出
    lines = cv2.HoughLines(horizontal, 1, np.pi / 180, 150)
    angles = []
    
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # θは0〜πの範囲。水平線はθが約0度またはπに近いが、ここでは基準を-90度からのずれで評価する
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)
    
    # 水平線が検出できなかった場合は回転処理を行わず元画像を返す
    if not angles:
        return image
    
    # 複数の線分から角度の中央値を算出（外れ値の影響を低減）
    median_angle = np.median(angles)
    
    # ほぼ水平の場合は、回転処理をスキップする（閾値は1度未満とする）
    if abs(median_angle) < 1.0:
        return image
    
    # 画像の回転処理：画像中心を基準に補正角度で回転
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(gray, rot_matrix, (w, h),
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_image_for_line_detection(image):
    """
    罫線検出のための画像前処理関数
    
    Parameters
    ----------
    image : numpy.ndarray
        処理対象のグレースケール画像
        
    Returns
    -------
    numpy.ndarray
        前処理された画像
    """
    # ノイズ除去のためのガウシアンブラー
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # ヒストグラム平坦化によるコントラスト強調
    equalized = cv2.equalizeHist(blurred)
    
    # シャープ化フィルタでエッジを強調
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(equalized, -1, kernel)
    
    return sharpened

def detect_lines_improved(image, adjust_for_image_size=True):
    """
    高品質な罫線検出関数（改良版）
    
    Parameters
    ----------
    image : numpy.ndarray
        処理対象のグレースケール画像
    adjust_for_image_size : bool, optional
        画像サイズに応じてパラメータを調整するか, by default True
        
    Returns
    -------
    tuple
        table_lines: 検出された罫線を含むバイナリ画像
        horizontal_lines: 水平方向の罫線のバイナリ画像
        vertical_lines: 垂直方向の罫線のバイナリ画像
    """
    # 前処理を適用
    preprocessed_image = preprocess_image_for_line_detection(image)
    
    height, width = preprocessed_image.shape
    
    # 画像サイズに応じたパラメータ調整
    if adjust_for_image_size:
        min_line_length = max(width, height) // 20
        size_factor = min(width, height) / 1000
        adaptive_kernel = max(1, int(31 * size_factor))
        if adaptive_kernel % 2 == 0:
            adaptive_kernel += 1  # 奇数にする
    else:
        min_line_length = 50
        adaptive_kernel = 31
    
    # 適応的二値化（局所的な照明条件に対応）
    binary = cv2.adaptiveThreshold(
        preprocessed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, adaptive_kernel, 11
    )
    
    # 水平・垂直方向の罫線を別々に検出
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_length, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_length))
    
    # 水平罫線の抽出
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    horizontal_lines = cv2.dilate(horizontal_lines, np.ones((1, 5), np.uint8), iterations=1)
    
    # 垂直罫線の抽出
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    vertical_lines = cv2.dilate(vertical_lines, np.ones((5, 1), np.uint8), iterations=1)
    
    # 罫線の連結性を向上（欠損部分の修復）
    h_dilate = cv2.dilate(horizontal_lines, np.ones((1, 21), np.uint8), iterations=1)
    v_dilate = cv2.dilate(vertical_lines, np.ones((21, 1), np.uint8), iterations=1)
    
    # 修復した罫線を元の幅に戻す
    h_erode = cv2.erode(h_dilate, np.ones((1, 15), np.uint8), iterations=1)
    v_erode = cv2.erode(v_dilate, np.ones((15, 1), np.uint8), iterations=1)
    
    # 最終的な罫線画像
    horizontal_lines = h_erode
    vertical_lines = v_erode
    
    # 水平・垂直罫線の統合
    table_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    # 罫線のフィルタリング：一定の長さ以上のものだけを保持
    min_line_length_pixels = min_line_length * 0.8  # 罫線の最小長さを調整
    horizontal_lines = filter_short_lines(horizontal_lines, min_line_length_pixels, axis=0)
    vertical_lines = filter_short_lines(vertical_lines, min_line_length_pixels, axis=1)
    
    return table_lines, horizontal_lines, vertical_lines

def filter_short_lines(lines_image, min_length, axis):
    """
    短い罫線をフィルタリングする関数
    
    Parameters
    ----------
    lines_image : numpy.ndarray
        罫線を含むバイナリ画像
    min_length : int
        最小の罫線長
    axis : int
        0 for horizontal, 1 for vertical
        
    Returns
    -------
    numpy.ndarray
        フィルタリングされた罫線画像
    """
    # 各行または列の白ピクセル数をカウント
    if axis == 0:  # horizontal
        sums = np.sum(lines_image, axis=1)
    else:  # vertical
        sums = np.sum(lines_image, axis=0)
    
    # 短い罫線を除去
    for i, sum_value in enumerate(sums):
        if sum_value < min_length:
            if axis == 0:
                lines_image[i, :] = 0
            else:
                lines_image[:, i] = 0
    
    return lines_image

def find_grid_intersections(horizontal_lines, vertical_lines):
    """
    水平線と垂直線の交点を検出し、グリッド構造の交点リストを返す関数

    Parameters
    ----------
    horizontal_lines : numpy.ndarray
        水平方向の罫線のバイナリ画像
    vertical_lines : numpy.ndarray
        垂直方向の罫線のバイナリ画像

    Returns
    -------
    list
        交点の座標リスト [(x, y), ...]
    """
    # 水平線と垂直線のビット積で交点を検出
    intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)
    
    # 交点の座標を取得
    intersection_points = []
    contours, _ = cv2.findContours(intersections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # 輪郭の重心を交点として記録
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            intersection_points.append((cx, cy))
    
    return intersection_points

def extract_line_positions(lines_image, axis='horizontal', min_length=20):
    """
    罫線画像から実際の水平/垂直線の位置を抽出する関数

    Parameters
    ----------
    lines_image : numpy.ndarray
        水平または垂直方向の罫線を含むバイナリ画像
    axis : str, optional
        抽出する線の方向 ('horizontal' または 'vertical'), by default 'horizontal'
    min_length : int, optional
        考慮する最小線長, by default 20

    Returns
    -------
    list
        検出された線の位置リスト
    """
    height, width = lines_image.shape
    positions = []
    
    if axis == 'horizontal':
        # 各行での白ピクセル数をカウント
        row_sums = np.sum(lines_image, axis=1)
        
        for y in range(height):
            # 一定以上の長さの線のみ考慮
            if row_sums[y] >= min_length:
                positions.append(y)
    else:  # vertical
        # 各列での白ピクセル数をカウント
        col_sums = np.sum(lines_image, axis=0)
        
        for x in range(width):
            # 一定以上の長さの線のみ考慮
            if col_sums[x] >= min_length:
                positions.append(x)
    
    # 近接した線の位置をマージ
    merged_positions = []
    if positions:
        positions.sort()
        current_group = [positions[0]]
        
        for pos in positions[1:]:
            # 前の位置との差が小さい場合は同じグループ
            if pos - current_group[-1] <= 3:
                current_group.append(pos)
            else:
                # グループの平均値を代表値として記録
                merged_positions.append(int(sum(current_group) / len(current_group)))
                current_group = [pos]
        
        # 最後のグループを処理
        if current_group:
            merged_positions.append(int(sum(current_group) / len(current_group)))
    
    return merged_positions

def detect_merged_cells(horizontal_positions, vertical_positions, horizontal_lines, vertical_lines, min_gap=10):
    """
    結合セルを検出する関数 - 罫線の有無に基づいて結合セルを識別

    Parameters
    ----------
    horizontal_positions : list
        水平線の位置リスト
    vertical_positions : list
        垂直線の位置リスト
    horizontal_lines : numpy.ndarray
        水平方向の罫線のバイナリ画像
    vertical_lines : numpy.ndarray
        垂直方向の罫線のバイナリ画像
    min_gap : int, optional
        セル間の最小ギャップ, by default 10

    Returns
    -------
    list
        検出されたセルの座標リスト [(x, y, w, h), ...]
    """
    height, width = horizontal_lines.shape
    cells = []
    
    # 各行ごとに処理
    for i in range(len(horizontal_positions) - 1):
        row_start = horizontal_positions[i]
        row_end = horizontal_positions[i + 1]
        
        # 行の高さが最小ギャップより小さい場合はスキップ
        if row_end - row_start < min_gap:
            continue
        
        # 各列ごとに処理
        j = 0
        while j < len(vertical_positions) - 1:
            col_start = vertical_positions[j]
            col_end = vertical_positions[j + 1]
            
            # 列の幅が最小ギャップより小さい場合はスキップ
            if col_end - col_start < min_gap:
                j += 1
                continue
            
            # セルの領域を設定
            cell_x = col_start
            cell_y = row_start
            cell_width = col_end - col_start
            cell_height = row_end - row_start
            
            # このセルの右側に結合セルがあるか確認（垂直線がない場合）
            next_col = j + 1
            while next_col < len(vertical_positions) - 1:
                # セル間の垂直線領域を抽出
                mid_x = (vertical_positions[next_col] + vertical_positions[next_col + 1]) // 2
                vert_line_region = vertical_lines[row_start:row_end, vertical_positions[next_col] - 2:vertical_positions[next_col] + 3]
                
                # 垂直線が存在しない場合（白ピクセルが少ない）
                if np.sum(vert_line_region) < (row_end - row_start) * 0.3:
                    # 結合セルとして幅を拡張
                    cell_width = vertical_positions[next_col + 1] - cell_x
                    next_col += 1
                else:
                    break
            
            # このセルの下側に結合セルがあるか確認（水平線がない場合）
            next_row = i + 1
            while next_row < len(horizontal_positions) - 1:
                # セル間の水平線領域を抽出
                horz_line_region = horizontal_lines[horizontal_positions[next_row] - 2:horizontal_positions[next_row] + 3, cell_x:cell_x + cell_width]
                
                # 水平線が存在しない場合（白ピクセルが少ない）
                if np.sum(horz_line_region) < (cell_width) * 0.3:
                    # 結合セルとして高さを拡張
                    cell_height = horizontal_positions[next_row + 1] - cell_y
                    next_row += 1
                else:
                    break
            
            # セルを追加
            cells.append((cell_x, cell_y, cell_width, cell_height))
            
            # 次の列へ進む（結合セルの分だけスキップ）
            j = next_col
    
    return cells

def extract_cells_from_actual_lines(image):
    """
    実際の罫線からセルを抽出する関数
    
    Parameters
    ----------
    image : numpy.ndarray
        処理対象のグレースケール画像
        
    Returns
    -------
    list
        検出されたセルの座標リスト [(x, y, w, h), ...]
    """
    # 罫線検出
    table_lines, h_lines, v_lines = detect_lines_improved(image)
    
    # 交点検出
    intersections = extract_intersections(h_lines, v_lines)
    
    # 交点からセルを構築
    cells = construct_cells_from_intersections(intersections, image.shape)
    
    # セルデータを検証・修正
    return validate_and_fix_cells(cells)

def validate_and_fix_cells(cells):
    """
    セルデータを検証し、必要に応じて修正する関数
    
    Parameters
    ----------
    cells : list or ndarray
        検証するセルデータ
        
    Returns
    -------
    list
        検証・修正済みのセルデータ [(x, y, w, h), ...]
    """
    if cells is None:
        return []
    
    # NumPy配列の場合は処理方法を変更
    if isinstance(cells, np.ndarray):
        # 2次元配列で、各行が座標を表す場合
        if len(cells.shape) == 2 and cells.shape[1] >= 4:
            return [(int(cell[0]), int(cell[1]), int(cell[2]), int(cell[3])) for cell in cells]
        # 1次元配列で長さが4の場合
        elif len(cells.shape) == 1 and cells.shape[0] == 4:
            return [(int(cells[0]), int(cells[1]), int(cells[2]), int(cells[3]))]
        else:
            print(f"警告: 不正なNumPy配列形式です: {cells.shape}")
            return []
    
    valid_cells = []
    
    for cell in cells:
        # タプルまたはリストの場合
        if isinstance(cell, (tuple, list)):
            if len(cell) >= 4:
                # 最初の4つの値を使用
                try:
                    x, y, w, h = cell[:4]
                    # 数値に変換できることを確認
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    valid_cells.append((x, y, w, h))
                except (ValueError, TypeError):
                    print(f"警告: セルの値が数値に変換できません: {cell[:4]}")
            else:
                print(f"警告: セルの要素数が不足しています: {cell}")
        # NumPy配列の場合
        elif isinstance(cell, np.ndarray):
            if len(cell.shape) == 1 and cell.shape[0] >= 4:
                try:
                    x, y, w, h = cell[:4]
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    valid_cells.append((x, y, w, h))
                except (ValueError, TypeError):
                    print(f"警告: NumPy配列の値が数値に変換できません: {cell[:4]}")
            else:
                print(f"警告: 不正なNumPy配列形式です: {cell.shape}")
        else:
            print(f"警告: 不明なセル形式です: {type(cell)}")
    
    return valid_cells

def construct_grid_structure(intersection_points, tolerance=10):
    """
    交点座標から表のグリッド構造（行と列の位置）を推定する関数

    Parameters
    ----------
    intersection_points : list
        交点の座標リスト [(x, y), ...]
    tolerance : int, optional
        同じ行/列とみなす座標の許容差, by default 10

    Returns
    -------
    tuple
        rows: 行の代表y座標のリスト (昇順)
        cols: 列の代表x座標のリスト (昇順)
    """
    if not intersection_points:
        return [], []
    
    # x座標とy座標を別々のリストとして抽出
    x_coords = [point[0] for point in intersection_points]
    y_coords = [point[1] for point in intersection_points]
    
    # クラスタリングによる行の識別
    rows = []
    y_coords.sort()
    
    # 初期値設定
    current_row = y_coords[0]
    rows.append(current_row)
    
    for y in y_coords:
        # 前の行と十分に離れている場合、新しい行として登録
        if y - current_row > tolerance:
            current_row = y
            rows.append(current_row)
    
    # クラスタリングによる列の識別
    cols = []
    x_coords.sort()
    
    # 初期値設定
    current_col = x_coords[0]
    cols.append(current_col)
    
    for x in x_coords:
        # 前の列と十分に離れている場合、新しい列として登録
        if x - current_col > tolerance:
            current_col = x
            cols.append(current_col)
    
    return rows, cols

def extract_cells_from_grid(rows, cols, min_size=20):
    """
    グリッド構造から表のセル領域を抽出する関数

    Parameters
    ----------
    rows : list
        行の代表y座標のリスト
    cols : list
        列の代表x座標のリスト
    min_size : int, optional
        セルとして認識する最小サイズ, by default 20

    Returns
    -------
    list
        セルの座標リスト [(x, y, w, h), ...]
    """
    cells = []
    
    # グリッドの行と列から全てのセルを生成
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            x = cols[j]
            y = rows[i]
            w = cols[j + 1] - cols[j]
            h = rows[i + 1] - rows[i]
            
            # 極端に小さいセルを除外
            if w > min_size and h > min_size:
                cells.append((x, y, w, h))
    
    return cells

def extract_cells_by_contours(lines_image, min_size=20):
    """
    輪郭ベースのアプローチでセルを検出する関数
    グリッド構造の検出が難しい場合のバックアップとして使用

    Parameters
    ----------
    lines_image : numpy.ndarray
        罫線を含むバイナリ画像
    min_size : int, optional
        セルとして認識する最小サイズ, by default 20

    Returns
    -------
    list
        セルの座標リスト [(x, y, w, h), ...]
    """
    # 罫線画像を膨張させてセルを閉じる
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(lines_image, kernel, iterations=2)
    
    # 反転して白黒を入れ替え（セルが白、背景が黒に）
    inverted = cv2.bitwise_not(dilated)
    
    # 輪郭を検出
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # セルとして適切なサイズの輪郭のみを抽出
    cells = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # 最小サイズ以上のみを考慮
        if w > min_size and h > min_size:
            cells.append((x, y, w, h))
    
    return cells

def draw_cells_on_image(original, cells):
    """
    セル領域を検出後の画像に描画する関数（異なる色を使用し、検出番号を追加）

    Parameters
    ----------
    original : numpy.ndarray
        元の画像（カラー）または二値化画像
    cells : list
        セルの座標リスト [(x, y, w, h), ...]

    Returns
    -------
    numpy.ndarray
        セルが描画された画像
    """
    # 元画像がグレースケールの場合、カラーに変換
    if len(original.shape) == 2:
        result = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        result = original.copy()

    for i, (x, y, w, h) in enumerate(cells):
        color = COLORS[i % len(COLORS)]  # 色を循環使用
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        # セル内部に検出番号を表示（視認しやすく）
        label_position = (x + 5, y + 20)
        cv2.putText(result, str(i + 1), label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return result

def draw_lines_on_image(original, lines):
    """
    罫線を検出後の画像に描画する関数（元のコードを保持）

    Parameters
    ----------
    original : numpy.ndarray
        元の画像（カラー）
    lines : numpy.ndarray
        検出された罫線を含むバイナリ画像

    Returns
    -------
    numpy.ndarray
        罫線が描画された画像
    """
    result = original.copy()
    edges = cv2.Canny(lines, 50, 150, apertureSize=3)
    detected_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=5)
    
    if detected_lines is not None:
        for i, line in enumerate(detected_lines):
            x1, y1, x2, y2 = line[0]
            color = COLORS[i % len(COLORS)]  # 10色を順番に使用
            cv2.line(result, (x1, y1), (x2, y2), color, 2)
            
            # 検出番号を描画（罫線の左上座標）
            label_position = (min(x1, x2) + 5, min(y1, y2) - 5)
            cv2.putText(result, str(i + 1), label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return result

def filter_cells_by_line_presence(cells, horizontal_lines, vertical_lines, min_line_ratio=0.3):
    """
    罫線の存在に基づいてセルをフィルタリングする関数
    
    Parameters
    ----------
    cells : list
        セルの座標リスト [(x, y, w, h), ...]
    horizontal_lines : numpy.ndarray
        水平方向の罫線のバイナリ画像
    vertical_lines : numpy.ndarray
        垂直方向の罫線のバイナリ画像
    min_line_ratio : float, optional
        セルの境界に必要な罫線の最小比率, by default 0.3
        
    Returns
    -------
    list
        フィルタリングされたセルの座標リスト
    """
    filtered_cells = []
    
    for cell in cells:
        # セルの要素数をチェック
        if len(cell) != 4:
            print(f"警告: セルの形式が不正です: {cell}")
            continue
            
        x, y, w, h = cell
        
        # セルの4辺に罫線が存在するか確認
        top_edge = horizontal_lines[y-2:y+3, x:x+w]
        bottom_edge = horizontal_lines[y+h-2:y+h+3, x:x+w]
        left_edge = vertical_lines[y:y+h, x-2:x+3]
        right_edge = vertical_lines[y:y+h, x+w-2:x+w+3]
        
        # 各辺の罫線の存在比率を計算
        top_ratio = np.sum(top_edge) / (w * 5) if w > 0 else 0
        bottom_ratio = np.sum(bottom_edge) / (w * 5) if w > 0 else 0
        left_ratio = np.sum(left_edge) / (h * 5) if h > 0 else 0
        right_ratio = np.sum(right_edge) / (h * 5) if h > 0 else 0
        
        # 少なくとも2辺以上に十分な罫線が存在する場合のみセルとして認識
        edge_count = sum(1 for ratio in [top_ratio, bottom_ratio, left_ratio, right_ratio] if ratio > min_line_ratio)
        
        if edge_count >= 2:
            filtered_cells.append((x, y, w, h))
    
    return filtered_cells

def merge_overlapping_cells(cells, overlap_threshold=0.5):
    """
    重複するセルをマージする関数
    
    Parameters
    ----------
    cells : list
        セルの座標リスト [(x, y, w, h), ...]
    overlap_threshold : float, optional
        重複とみなす面積の閾値, by default 0.5
        
    Returns
    -------
    list
        マージされたセルの座標リスト
    """
    if not cells:
        return []
    
    # セルを面積の降順にソート
    sorted_cells = sorted(cells, key=lambda c: c[2] * c[3], reverse=True)
    merged_cells = []
    
    for cell in sorted_cells:
        x1, y1, w1, h1 = cell
        should_add = True
        
        for i, merged_cell in enumerate(merged_cells):
            x2, y2, w2, h2 = merged_cell
            
            # 重複領域の計算
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = x_overlap * y_overlap
            
            # 小さいセルの面積に対する重複の割合
            smaller_area = min(w1 * h1, w2 * h2)
            if smaller_area > 0 and overlap_area / smaller_area > overlap_threshold:
                should_add = False
                break
        
        if should_add:
            merged_cells.append(cell)
    
    return merged_cells

def advanced_iterative_refinement(image, initial_cells, max_iterations=5):
    """
    高品質な反復チェック工程によるセル検出の改善関数（最適化版）
    
    Parameters
    ----------
    image : numpy.ndarray
        処理対象のグレースケール画像
    initial_cells : list
        初期検出されたセルの座標リスト
    max_iterations : int, optional
        最大反復回数, by default 5
        
    Returns
    -------
    tuple
        final_cells: 最終的に改善されたセルの座標リスト
        table_lines: 検出された罫線を含むバイナリ画像
        horizontal_lines: 水平方向の罫線のバイナリ画像
        vertical_lines: 垂直方向の罫線のバイナリ画像
        iteration_history: 各反復での検出結果の履歴
    """
    # 罫線検出
    table_lines, horizontal_lines, vertical_lines = detect_lines_improved(image)
    
    # 水平線と垂直線の位置を抽出
    h_positions = extract_line_positions(horizontal_lines, 'horizontal')
    v_positions = extract_line_positions(vertical_lines, 'vertical')
    
    # 反復履歴の初期化
    iteration_history = {
        'cells': [],
        'metrics': []
    }
    
    # 初期セルのリストをコピー（tupleの場合はリストに変換）
    if isinstance(initial_cells, tuple):
        current_cells = list(initial_cells)
    else:
        current_cells = initial_cells.copy() if hasattr(initial_cells, 'copy') else list(initial_cells)
    
    # 反復改善プロセス
    for iteration in range(max_iterations):
        # ステップ1: 罫線存在率に基づくセルのフィルタリング
        filtered_cells = filter_cells_by_line_presence(
            current_cells, horizontal_lines, vertical_lines, 
            min_line_ratio=0.15  # より緩和された閾値
        )
        
        # フィルタリング後にセルがなくなった場合は前の結果を維持
        if not filtered_cells:
            filtered_cells = current_cells
        
        # ステップ2: グリッド構造との整合性チェック
        grid_aligned_cells = align_to_grid_structure(
            filtered_cells, h_positions, v_positions
        )
        
        # 整列後にセルがなくなった場合は前の結果を維持
        if not grid_aligned_cells:
            grid_aligned_cells = filtered_cells
        
        # ステップ3: 重複セルの統合と無効セルの除去
        # 問題修正: 厳しすぎる条件を緩和
        merged_cells = remove_invalid_cells(
            grid_aligned_cells, table_lines, 
            min_size=10, max_overlap=0.8  # 緩和された条件
        )
        
        # 統合後にセルがなくなった場合は前の結果を維持
        if not merged_cells:
            merged_cells = grid_aligned_cells
        
        # ステップ4: 結合セルの検出と追加
        with_merged_cells = detect_and_add_merged_cells(
            merged_cells, horizontal_lines, vertical_lines, 
            h_positions, v_positions
        )
        
        # ステップ5: 最終的なセルの整列とソート
        # エラー修正: セルの形式を確認してからソート
        valid_cells_for_sort = []
        for cell in with_merged_cells:
            if isinstance(cell, tuple) and len(cell) == 4:
                valid_cells_for_sort.append(cell)
            elif isinstance(cell, np.ndarray) and cell.shape == (4,):
                # NumPy配列をタプルに変換
                valid_cells_for_sort.append(tuple(cell))
            else:
                print(f"警告: セルの形式が不正です（ソート時）: {cell}")
        
        refined_cells = sorted(valid_cells_for_sort, key=lambda cell: (cell[1], cell[0]))
        
        # 品質メトリクスの計算
        quality_metrics = calculate_cell_quality_metrics(
            refined_cells, horizontal_lines, vertical_lines
        )
        
        # 履歴に追加
        iteration_history['cells'].append(refined_cells)
        iteration_history['metrics'].append(quality_metrics)
        
        # 現在のセル集合を更新
        current_cells = refined_cells
        
        # 品質が十分に高い場合、または改善が見られない場合は早期終了
        if iteration > 0:
            prev_metrics = iteration_history['metrics'][-2]
            curr_metrics = iteration_history['metrics'][-1]
            
            # 品質が十分に高い、または改善が少ない場合
            if (curr_metrics['overall_quality'] > 0.85 or  
                (curr_metrics['overall_quality'] - prev_metrics['overall_quality']) < 0.03):
                break
    
    # グリッドの各領域をチェック
    for i in range(len(h_positions) - 1):
        for j in range(len(v_positions) - 1):
            top = h_positions[i]
            bottom = h_positions[i + 1]
            left = v_positions[j]
            right = v_positions[j + 1]
            
            # この領域が既存のセルに含まれているかチェック
            is_covered = False
            for cell in cells:
                # セルの要素数をチェック
                if len(cell) != 4:
                    print(f"警告: セルの形式が不正です: {cell}")
                    continue
                
                x, y, w, h = cell
                if (x <= left and x + w >= right and 
                    y <= top and y + h >= bottom):
                    is_covered = True
                    break
            
            if is_covered:
                continue
            
            # 水平方向の結合セルをチェック
            h_merged = True
            for k in range(j + 1, len(v_positions) - 1):
                mid_x = (v_positions[k] + v_positions[k + 1]) // 2
                vert_line_region = v_lines[top:bottom, v_positions[k] - 2:v_positions[k] + 3]
                
                # 垂直線が存在する場合は結合セルではない
                if np.sum(vert_line_region) > (bottom - top) * 0.3:
                    h_merged = False
                    break
            
            # 垂直方向の結合セルをチェック
            v_merged = True
            for k in range(i + 1, len(h_positions) - 1):
                horz_line_region = h_lines[h_positions[k] - 2:h_positions[k] + 3, left:right]
                
                # 水平線が存在する場合は結合セルではない
                if np.sum(horz_line_region) > (right - left) * 0.3:
                    v_merged = False
                    break
            
            # 結合セルが検出された場合、追加
            if h_merged or v_merged:
                # 結合セルの範囲を決定
                merged_right = right
                merged_bottom = bottom
                
                if h_merged:
                    for k in range(j + 1, len(v_positions) - 1):
                        if np.sum(v_lines[top:bottom, v_positions[k] - 2:v_positions[k] + 3]) <= (bottom - top) * 0.3:
                            merged_right = v_positions[k + 1]
                        else:
                            break
                
                if v_merged:
                    for k in range(i + 1, len(h_positions) - 1):
                        if np.sum(h_lines[h_positions[k] - 2:h_positions[k] + 3, left:right]) <= (right - left) * 0.3:
                            merged_bottom = h_positions[k + 1]
                        else:
                            break
                
                # 結合セルを追加
                result_cells.append((left, top, merged_right - left, merged_bottom - top))
    
    return result_cells

def calculate_cell_quality_metrics(cells, h_lines, v_lines):
    """
    セル検出の品質メトリクスを計算する関数
    
    Parameters
    ----------
    cells : list
        セルの座標リスト [(x, y, w, h), ...]
    h_lines : numpy.ndarray
        水平方向の罫線のバイナリ画像
    v_lines : numpy.ndarray
        垂直方向の罫線のバイナリ画像
        
    Returns
    -------
    dict
        品質メトリクスを含む辞書
    """
    if not cells:
        return {'overall_quality': 0.0, 'line_alignment': 0.0, 'coverage': 0.0}
    
    # 罫線との整合性スコア
    line_alignment_scores = []
    for x, y, w, h in cells:
        # セルの4辺に罫線が存在するか確認
        top_edge = h_lines[y-2:y+3, x:x+w]
        bottom_edge = h_lines[y+h-2:y+h+3, x:x+w]
        left_edge = v_lines[y:y+h, x-2:x+3]
        right_edge = v_lines[y:y+h, x+w-2:x+w+3]
        
        # 各辺の罫線の存在比率を計算
        top_ratio = np.sum(top_edge) / (w * 5) if w > 0 else 0
        bottom_ratio = np.sum(bottom_edge) / (w * 5) if w > 0 else 0
        left_ratio = np.sum(left_edge) / (h * 5) if h > 0 else 0
        right_ratio = np.sum(right_edge) / (h * 5) if h > 0 else 0
        
        # セルの罫線整合性スコア
        cell_alignment = (top_ratio + bottom_ratio + left_ratio + right_ratio) / 4
        line_alignment_scores.append(cell_alignment)
    
    # 平均罫線整合性スコア
    avg_line_alignment = sum(line_alignment_scores) / len(line_alignment_scores) if line_alignment_scores else 0
    
    # 罫線カバレッジスコア（検出されたセルが罫線をどれだけカバーしているか）
    h, w = h_lines.shape
    coverage_mask = np.zeros((h, w), dtype=np.uint8)
    
    for x, y, w, h in cells:
        coverage_mask[y:y+h, x:x+w] = 1
    
    # 罫線画像と重なる部分
    covered_lines = cv2.bitwise_and(
        cv2.add(h_lines, v_lines), 
        coverage_mask * 255
    )
    
    # カバレッジスコア
    total_lines = np.sum(cv2.add(h_lines, v_lines))
    covered_lines_sum = np.sum(covered_lines)
    coverage_score = covered_lines_sum / total_lines if total_lines > 0 else 0
    
    # 総合品質スコア
    overall_quality = (avg_line_alignment * 0.6) + (coverage_score * 0.4)
    
    return {
        'overall_quality': overall_quality,
        'line_alignment': avg_line_alignment,
        'coverage': coverage_score
    }

def cluster_positions(positions, tolerance=10):
    """
    近接した位置をクラスタリングする関数
    
    Parameters
    ----------
    positions : list
        位置の座標リスト
    tolerance : int, optional
        同一クラスタとみなす最大距離, by default 10
        
    Returns
    -------
    list
        クラスタリングされた位置の座標リスト
    """
    if not positions:
        return []
    
    # 位置を昇順にソート
    sorted_positions = sorted(positions)
    
    # クラスタリング
    clusters = []
    current_cluster = [sorted_positions[0]]
    
    for pos in sorted_positions[1:]:
        # 前の位置との距離が閾値以内なら同じクラスタ
        if pos - current_cluster[-1] <= tolerance:
            current_cluster.append(pos)
        else:
            # 新しいクラスタを開始
            clusters.append(sum(current_cluster) // len(current_cluster))
            current_cluster = [pos]
    
    # 最後のクラスタを追加
    if current_cluster:
        clusters.append(sum(current_cluster) // len(current_cluster))
    
    return clusters

def detect_junction_points_enhanced(image):
    """
    複数の手法を組み合わせた高品質な接合部検出関数
    """
    # 前処理の強化
    preprocessed = preprocess_image_for_line_detection(image)
    binary = cv2.adaptiveThreshold(
        preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 31, 11
    )
    
    # ノイズ除去の強化
    binary = cv2.medianBlur(binary, 3)
    
    # 20x20サイズのカーネルを定義（より精密な形状）
    kernels = {
        'cross': np.zeros((20, 20), dtype=np.uint8),
        'l': np.zeros((20, 20), dtype=np.uint8),
        't': np.zeros((20, 20), dtype=np.uint8)
    }
    
    # 十字型（+）- より細かい線幅制御
    kernels['cross'][8:12, :] = 1  # 水平線
    kernels['cross'][:, 8:12] = 1  # 垂直線
    kernels['cross'] = cv2.GaussianBlur(kernels['cross'], (3, 3), 0)
    
    # L字型 - エッジ部分を滑らかに
    kernels['l'][8:12, 8:] = 1  # 水平線
    kernels['l'][8:, 8:12] = 1  # 垂直線
    kernels['l'] = cv2.GaussianBlur(kernels['l'], (3, 3), 0)
    
    # T字型 - より安定した検出のための形状
    kernels['t'][8:12, :] = 1  # 水平線
    kernels['t'][8:, 8:12] = 1  # 垂直線の一部
    kernels['t'] = cv2.GaussianBlur(kernels['t'], (3, 3), 0)
    
    # 1. 改良されたモルフォロジー演算による検出
    morphology_points = []
    for kernel_name, kernel in kernels.items():
        # マルチスケール検出
        for scale in [0.8, 1.0, 1.2]:
            scaled_kernel = cv2.resize(kernel, None, fx=scale, fy=scale)
            if scaled_kernel.shape[0] % 2 == 0:
                scaled_kernel = cv2.copyMakeBorder(scaled_kernel, 0, 1, 0, 1, cv2.BORDER_CONSTANT)
            
            points = cv2.morphologyEx(binary, cv2.MORPH_HITMISS, scaled_kernel.astype(np.uint8))
            coords = np.where(points > 0)
            for y, x in zip(coords[0], coords[1]):
                morphology_points.append((x, y))
    
    # 2. 改良されたテンプレートマッチング
    template_points = []
    for kernel_name, kernel in kernels.items():
        # マルチスケール検出の範囲を拡大し、より細かいステップで実行
        for scale in [0.6, 0.8, 1.0, 1.2, 1.4]:
            scaled_kernel = cv2.resize(kernel, None, fx=scale, fy=scale)
            if scaled_kernel.shape[0] % 2 == 0:
                scaled_kernel = cv2.copyMakeBorder(scaled_kernel, 0, 1, 0, 1, cv2.BORDER_CONSTANT)
            
            # カーネルの正規化を追加
            scaled_kernel = scaled_kernel.astype(np.float32)
            scaled_kernel = scaled_kernel / np.sum(scaled_kernel)
            
            # テンプレートマッチングの閾値を調整（0.9→0.75）
            result = cv2.matchTemplate(
                binary.astype(np.float32),
                scaled_kernel,
                cv2.TM_CCORR_NORMED
            )
            
            # 閾値を下げて検出感度を上げる
            coords = np.where(result >= 0.75)
            points = list(zip(coords[1], coords[0]))
            
            # 非最大値抑制の距離閾値を調整
            points = non_max_suppression(points, result, distance_threshold=15)
            
            for x, y in points:
                template_points.append((
                    x + scaled_kernel.shape[1]//2,
                    y + scaled_kernel.shape[0]//2
                ))
    
    # 3. 改良されたコサイン類似度検出
    cosine_points = []
    # カーネルベクトルの正規化を改善
    kernel_vectors = []
    for kernel in kernels.values():
        kernel_flat = kernel.flatten()
        if np.sum(kernel_flat) > 0:  # ゼロ除算を防ぐ
            kernel_vectors.append(kernel_flat / np.linalg.norm(kernel_flat))
    
    # スライディングウィンドウのパラメータを調整
    step_size = 3  # より細かいステップサイズ
    window_size = 20
    
    for y in range(0, binary.shape[0] - window_size, step_size):
        for x in range(0, binary.shape[1] - window_size, step_size):
            window = binary[y:y+window_size, x:x+window_size].flatten()
            window_sum = np.sum(window)
            
            # 非ゼロ領域のみを処理
            if window_sum > 0:
                window_norm = window / (np.linalg.norm(window) + 1e-6)
                
                # 各カーネルとの類似度を計算
                similarities = [np.dot(window_norm, kv) for kv in kernel_vectors]
                max_similarity = max(similarities)
                
                # 閾値を調整（0.9→0.75）
                if max_similarity >= 0.75:
                    cosine_points.append((x + window_size//2, y + window_size//2))
    
    # 重複点の除去と統合の距離閾値を調整
    morphology_points = remove_duplicate_points(morphology_points, distance_threshold=15)
    template_points = remove_duplicate_points(template_points, distance_threshold=15)
    cosine_points = remove_duplicate_points(cosine_points, distance_threshold=15)
    
    # 検出結果の可視化（改良版）
    visualization_images = []
    for points, color in [
        (morphology_points, (0, 0, 255)),
        (template_points, (0, 255, 0)),
        (cosine_points, (255, 0, 0))
    ]:
        vis_img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        
        # 検出点の描画を改良
        for x, y in points:
            # 中心点
            cv2.circle(vis_img, (x, y), 5, color, -1)
            # 検出点の周囲に円を描画
            cv2.circle(vis_img, (x, y), 7, color, 1)
        
        visualization_images.append(vis_img)
    
    return (morphology_points, template_points, cosine_points,
            *visualization_images)

def non_max_suppression(points, scores, distance_threshold):
    """
    非最大値抑制による重複検出の除去
    
    Parameters
    ----------
    points : list
        検出点の座標リスト [(x, y), ...]
    scores : numpy.ndarray
        各点のスコア
    distance_threshold : int
        重複とみなす距離の閾値
        
    Returns
    -------
    list
        非最大値抑制後の点のリスト
    """
    if not points:
        return []
    
    # スコアでソート
    points_with_scores = [
        (x, y, scores[y, x]) for x, y in points
    ]
    points_with_scores.sort(key=lambda x: x[2], reverse=True)
    
    selected = []
    for x1, y1, s1 in points_with_scores:
        should_select = True
        for x2, y2, _ in selected:
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < distance_threshold:
                should_select = False
                break
        if should_select:
            selected.append((x1, y1, s1))
    
    return [(x, y) for x, y, _ in selected]

def remove_duplicate_points(points, distance_threshold):
    """
    近接した点を統合する関数
    
    Parameters
    ----------
    points : list
        点の座標リスト [(x, y), ...]
    distance_threshold : int
        統合する点間の最大距離
        
    Returns
    -------
    list
        統合後の点のリスト
    """
    if not points:
        return []
    
    # 点をクラスタリング
    clusters = []
    for x, y in points:
        added_to_cluster = False
        for cluster in clusters:
            # クラスタ内の任意の点との距離を確認
            for cx, cy in cluster:
                distance = np.sqrt((x - cx)**2 + (y - cy)**2)
                if distance < distance_threshold:
                    cluster.append((x, y))
                    added_to_cluster = True
                    break
            if added_to_cluster:
                break
        
        if not added_to_cluster:
            clusters.append([(x, y)])
    
    # 各クラスタの重心を計算
    merged_points = []
    for cluster in clusters:
        x_mean = int(np.mean([x for x, _ in cluster]))
        y_mean = int(np.mean([y for _, y in cluster]))
        merged_points.append((x_mean, y_mean))
    
    return merged_points

def reconstruct_table_from_junctions(image, junction_points):
    """
    接合部から表構造を再構築する関数（改良版）
    
    Parameters
    ----------
    image : numpy.ndarray
        元の画像
    junction_points : list
        接合部の座標と種類のリスト [(x, y, type), ...]
        
    Returns
    -------
    tuple
        reconstructed_cells: 再構築されたセルの座標リスト
        reconstructed_image: 再構築された表の可視化画像
    """
    height, width = image.shape[:2]
    reconstructed_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 接合部の座標のみを抽出
    junction_coords = [(x, y) for x, y, _ in junction_points]
    
    # 接合部をグリッドに整列
    h_positions = cluster_positions([y for _, y, _ in junction_points], tolerance=15)
    v_positions = cluster_positions([x for x, _, _ in junction_points], tolerance=15)
    
    # グリッド線を描画
    for y in h_positions:
        cv2.line(reconstructed_image, (0, y), (width, y), (0, 255, 0), 1)
    
    for x in v_positions:
        cv2.line(reconstructed_image, (x, 0), (x, height), (0, 255, 0), 1)
    
    # 接合部を描画
    for x, y, j_type in junction_points:
        cv2.circle(reconstructed_image, (x, y), 3, (0, 0, 255), -1)  # 赤色
    
    # セルの再構築
    reconstructed_cells = []
    
    # 各水平位置ペアについて
    for i in range(len(h_positions) - 1):
        y1 = h_positions[i]
        y2 = h_positions[i+1]
        
        # 各垂直位置ペアについて
        for j in range(len(v_positions) - 1):
            x1 = v_positions[j]
            x2 = v_positions[j+1]
            
            # 矩形の4つの角の座標
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            
            # 各角に最も近い接合部を見つける
            corner_junctions = []
            for corner in corners:
                # 接合部が存在する場合
                if junction_coords:
                    # 最も近い接合部を見つける
                    closest_junction = min(junction_coords, key=lambda j: np.sqrt((j[0]-corner[0])**2 + (j[1]-corner[1])**2))
                    
                    # 距離が閾値以内なら採用
                    dist = np.sqrt((closest_junction[0]-corner[0])**2 + (closest_junction[1]-corner[1])**2)
                    if dist < 20:  # 閾値を調整
                        corner_junctions.append(closest_junction)
            
            # 少なくとも3つの角に接合部が存在する場合、セルとして認識
            if len(corner_junctions) >= 3:
                # 左上と右下の座標を使ってセルを定義
                min_x = min(j[0] for j in corner_junctions)
                min_y = min(j[1] for j in corner_junctions)
                max_x = max(j[0] for j in corner_junctions)
                max_y = max(j[1] for j in corner_junctions)
                
                # セルの幅と高さが十分な場合のみ追加
                if max_x - min_x > 20 and max_y - min_y > 20:
                    reconstructed_cells.append((min_x, min_y, max_x - min_x, max_y - min_y))
                    cv2.rectangle(reconstructed_image, (min_x, min_y), (max_x, max_y), (255, 0, 255), 1)
    
    return reconstructed_cells, reconstructed_image

def process_table_image_enhanced(image_path):
    """
    拡張版の表画像処理関数
    
    Parameters
    ----------
    image_path : str
        処理する画像のパス
    """
    # 画像の読み込みと前処理
    img, gray = load_image(image_path)
    corrected = correct_skew(gray)
    
    # 各手法による接合部検出
    junctions, template_junctions, cosine_junctions, \
    morph_image, template_image, cosine_image = \
        detect_junction_points_enhanced(corrected)
    
    # 各手法でのセル再構築
    morph_cells, morph_result = reconstruct_table_from_junctions(corrected, 
        [(x, y, 'point') for x, y in junctions])
    
    template_cells, template_result = reconstruct_table_from_junctions(corrected,
        [(x, y, 'point') for x, y in template_junctions])
    
    cosine_cells, cosine_result = reconstruct_table_from_junctions(corrected,
        [(x, y, 'point') for x, y in cosine_junctions])
    
    # 結果の可視化（3行4列）
    plt.figure(figsize=(20, 15))
    
    # フォントの設定
    plt.rcParams['font.family'] = 'MS Gothic'  # 日本語フォントの設定
    
    # 1行目：元画像と前処理
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("入力画像")
    plt.axis("off")
    
    plt.subplot(3, 4, 2)
    plt.imshow(corrected, cmap='gray')
    plt.title("傾き補正済み画像")
    plt.axis("off")
    
    plt.subplot(3, 4, 3)
    plt.imshow(cv2.cvtColor(morph_image, cv2.COLOR_BGR2RGB))
    plt.title("モルフォロジー処理による\n接合点検出結果")
    plt.axis("off")
    
    plt.subplot(3, 4, 4)
    plt.imshow(cv2.cvtColor(morph_result, cv2.COLOR_BGR2RGB))
    plt.title("モルフォロジー処理による\nセル検出結果")
    plt.axis("off")
    
    # 2行目：テンプレートマッチング結果
    plt.subplot(3, 4, 5)
    plt.imshow(cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB))
    plt.title("テンプレートマッチングによる\n接合点検出結果")
    plt.axis("off")
    
    plt.subplot(3, 4, 6)
    plt.imshow(cv2.cvtColor(template_result, cv2.COLOR_BGR2RGB))
    plt.title("テンプレートマッチングによる\nセル検出結果")
    plt.axis("off")
    
    # 3行目：コサイン類似度結果
    plt.subplot(3, 4, 7)
    plt.imshow(cv2.cvtColor(cosine_image, cv2.COLOR_BGR2RGB))
    plt.title("コサイン類似度による\n接合点検出結果")
    plt.axis("off")
    
    plt.subplot(3, 4, 8)
    plt.imshow(cv2.cvtColor(cosine_result, cv2.COLOR_BGR2RGB))
    plt.title("コサイン類似度による\nセル検出結果")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def construct_cells_from_intersections(intersections, image_shape, min_cell_size=20):
    """
    交点からセルを構築する関数（改良版）
    
    Parameters
    ----------
    intersections : list
        交点の座標リスト [(x, y), ...]
    image_shape : tuple
        元画像の形状 (height, width)
    min_cell_size : int, optional
        最小セルサイズ, by default 20
        
    Returns
    -------
    list
        セルの座標リスト [(x, y, w, h), ...]
    """
    if len(intersections) < 4:  # 少なくとも4つの交点が必要
        return []
    
    # 交点をx座標とy座標でソート
    x_sorted = sorted(set([x for x, _ in intersections]))
    y_sorted = sorted(set([y for _, y in intersections]))
    
    # 近接した座標をクラスタリング
    x_clusters = cluster_positions(x_sorted, tolerance=15)
    y_clusters = cluster_positions(y_sorted, tolerance=15)
    
    # 交点の存在を確認するための2次元グリッド
    grid = np.zeros((len(y_clusters), len(x_clusters)), dtype=bool)
    
    # 各交点を最も近いグリッド位置に割り当て
    for x, y in intersections:
        # 最も近いx_clusterとy_clusterのインデックスを見つける
        x_idx = min(range(len(x_clusters)), key=lambda i: abs(x_clusters[i] - x))
        y_idx = min(range(len(y_clusters)), key=lambda i: abs(y_clusters[i] - y))
        grid[y_idx, x_idx] = True
    
    # セルの構築
    cells = []
    
    # グリッドを走査して隣接する交点からセルを形成
    for i in range(len(y_clusters) - 1):
        for j in range(len(x_clusters) - 1):
            # セルの4つの角が存在するか確認
            top_left = grid[i, j]
            top_right = grid[i, j+1]
            bottom_left = grid[i+1, j]
            bottom_right = grid[i+1, j+1]
            
            # 少なくとも3つの角が存在する場合、セルとして認識
            corner_count = sum([top_left, top_right, bottom_left, bottom_right])
            if corner_count >= 3:
                x = x_clusters[j]
                y = y_clusters[i]
                w = x_clusters[j+1] - x
                h = y_clusters[i+1] - y
                
                # 最小サイズ以上のセルのみを追加
                if w >= min_cell_size and h >= min_cell_size:
                    cells.append((x, y, w, h))
    
    # 結合セルの検出（隣接するセルが存在しない場合）
    for i in range(len(y_clusters) - 2):
        for j in range(len(x_clusters) - 1):
            # 水平方向の結合セルをチェック
            if (grid[i, j] and grid[i, j+1] and 
                grid[i+2, j] and grid[i+2, j+1] and 
                not grid[i+1, j] and not grid[i+1, j+1]):
                
                x = x_clusters[j]
                y = y_clusters[i]
                w = x_clusters[j+1] - x
                h = y_clusters[i+2] - y
                
                if w >= min_cell_size and h >= min_cell_size:
                    cells.append((x, y, w, h))
    
    for i in range(len(y_clusters) - 1):
        for j in range(len(x_clusters) - 2):
            # 垂直方向の結合セルをチェック
            if (grid[i, j] and grid[i+1, j] and 
                grid[i, j+2] and grid[i+1, j+2] and 
                not grid[i, j+1] and not grid[i+1, j+1]):
                
                x = x_clusters[j]
                y = y_clusters[i]
                w = x_clusters[j+2] - x
                h = y_clusters[i+1] - y
                
                if w >= min_cell_size and h >= min_cell_size:
                    cells.append((x, y, w, h))
    
    return cells

# メイン処理で呼び出し
image_path = r"D:\winpythons\WPy64-31180_rinna\venv\app_rakurakuOCR\src\rakurakuOCR_functions\dataset\table_a_01_angleR5.png"
process_table_image_enhanced(image_path)