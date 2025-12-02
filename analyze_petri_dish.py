import cv2
import matplotlib.pyplot as plt
import numpy as np


def align_images(img_ref, img_target):
    """
    位置・サイズ・角度合わせ (AKAZE)
    """
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(gray_ref, None)
    kp2, des2 = akaze.detectAndCompute(gray_target, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 精度確保のため上位15%を利用
    good_matches = matches[:int(len(matches) * 0.15)]

    if len(good_matches) < 4:
        h, w = img_ref.shape[:2]
        return cv2.resize(img_target, (w, h))

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = img_ref.shape[:2]
    return cv2.warpPerspective(img_target, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def analyze_petri_dish_save_result(path_before, path_after):
    # 画像読み込み
    img_before = cv2.imread(path_before)
    img_after = cv2.imread(path_after)

    if img_before is None or img_after is None:
        print("画像が見つかりません。パスを確認してください。")
        return

    # 1. 位置合わせ
    print("位置合わせを実行中...")
    img_after_aligned = align_images(img_before, img_after)

    # 2. オプティカルフロー計算
    gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(img_after_aligned, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev=gray_after, next=gray_before, flow=None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0
    )

    # Warping (Fitting)
    h, w = gray_before.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)

    warped_before = cv2.remap(img_before, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # 3. 差分の計算
    diff = cv2.absdiff(img_after_aligned, warped_before)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # ノイズ除去（閾値を少し高めに設定して細かいゴミを無視）
    _, diff_thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

    # 4. マージ処理（赤み抑制版）
    b, g, r = cv2.split(warped_before)

    # --- 調整エリア ---
    # 赤みの強さ係数（1.0〜2.0くらいが自然。前回は3.0でした）
    red_intensity = 1.5

    # 差分を増幅させる
    diff_boost = cv2.multiply(diff_gray, red_intensity)

    # Rチャネルには加算（赤くする）
    r_new = cv2.add(r, diff_boost)

    # BとGチャネルからは少しだけ減算する
    # ※白背景の場合、Rを足しても白(255,255,255)のまま変わらないため、
    #   BとGを少し下げることで相対的に赤く見せます。
    #   ただし、元画像を残すため減算係数は小さく(0.5)しています。
    diff_subtract = cv2.multiply(diff_gray, red_intensity * 0.5)

    # 型を合わせるために変換してから計算
    b_new = cv2.subtract(b, diff_subtract.astype(np.uint8))
    g_new = cv2.subtract(g, diff_subtract.astype(np.uint8))

    merged_result = cv2.merge((b_new, g_new, r_new))
    # ------------------

    # 5. 画像の保存
    cv2.imwrite('diff_image.png', diff_gray)
    cv2.imwrite('merged_result.png', merged_result)
    print("画像を保存しました: diff_image.png, merged_result.png")

    # 6. 表示
    titles = ['Fitted Before', 'Aligned After', 'Diff (Saved)', 'Result (Saved)']
    images = [
        cv2.cvtColor(warped_before, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(img_after_aligned, cv2.COLOR_BGR2RGB),
        diff_gray,
        cv2.cvtColor(merged_result, cv2.COLOR_BGR2RGB)
    ]

    plt.figure(figsize=(16, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        if i == 2:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()

    save_path = 'comparison_result_all.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"4枚組み合わせた画像を保存しました: {save_path}")

    plt.show()


# 実行
analyze_petri_dish_save_result('before.jpg', 'after.jpg')
