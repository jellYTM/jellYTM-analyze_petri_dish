import cv2
import matplotlib.pyplot as plt
import numpy as np


def align_images(img_ref, img_target):
    """
    特徴点マッチングを用いて img_target を img_ref の位置・サイズ・角度に合わせる関数
    """
    # グレースケール化
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

    # AKAZE検出器の準備
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(gray_ref, None)
    kp2, des2 = akaze.detectAndCompute(gray_target, None)

    # 特徴点マッチング (Brute-Force Hamming)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # マッチング精度の高い順にソートして上位を採用
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:int(len(matches) * 0.15)]  # 上位15%を利用

    if len(good_matches) < 4:
        print("特徴点が十分に検出できませんでした。単純リサイズを行います。")
        h, w = img_ref.shape[:2]
        return cv2.resize(img_target, (w, h))

    # 変換行列（ホモグラフィ）の推定
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 射影変換でimg_targetを変形させてimg_refに合わせる
    h, w = img_ref.shape[:2]
    aligned_img = cv2.warpPerspective(img_target, M, (w, h))

    return aligned_img


def analyze_petri_dish_robust(path_before, path_after):
    # 1. 画像読み込み
    img_before = cv2.imread(path_before)
    img_after = cv2.imread(path_after)

    if img_before is None or img_after is None:
        print("画像が見つかりません")
        return

    # ★ここで位置合わせとサイズ統一を同時に行う★
    print("位置合わせを実行中...")
    img_after_aligned = align_images(img_before, img_after)

    # 以降はオプティカルフロー処理（前回のコードと同じ流れ）
    gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(img_after_aligned, cv2.COLOR_BGR2GRAY)

    # Farneback オプティカルフロー
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

    # 差分可視化
    diff = cv2.absdiff(img_after_aligned, warped_before)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    diff_heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)

    # 表示
    titles = ['Original Before', 'Aligned After', 'Fitted Before (Flow)', 'Difference']
    images = [img_before, img_after_aligned, warped_before, diff_heatmap]

    plt.figure(figsize=(16, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        # BGR -> RGB変換して表示
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


analyze_petri_dish_robust('PXL_20251118_061920040_clipped.jpg', 'PXL_20251119_052741717_clipped.jpg')
