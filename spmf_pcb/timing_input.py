"""Fixed three-band input for the separate forward-timing experiment."""
import cv2
import numpy as np

def normalize_u8(img):
    img = img.astype(np.float32)
    p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
    img = np.clip(img, p2, p98)
    if p98 - p2 < 1e-6: return np.zeros_like(img, dtype=np.uint8)
    return ((img - p2) / (p98 - p2) * 255.0).astype(np.uint8)

def apply_transform(img, params, target_shape):
    h_t, w_t = target_shape[:2]
    h_s, w_s = img.shape[:2]

    # 1. 基础参数
    scale_x = params['sx'] / 1000.0
    scale_y = params['sy'] / 1000.0
    rotation = (params.get('rot', 500) - 500) / 10.0
    # 新增：超大平移范围 (2000中心)
    dx = params['dx'] - 2000
    dy = params['dy'] - 2000
    # 新增：透视参数 (500中心)
    px = (params.get('px', 500) - 500) / 10000.0  # 极微小透视
    py = (params.get('py', 500) - 500) / 10000.0

    if scale_x <= 0: scale_x = 0.001
    if scale_y <= 0: scale_y = 0.001

    # 2. 缩放
    new_w = int(w_s * scale_x)
    new_h = int(h_s * scale_y)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 3. 旋转 + 透视 (合并为 Homography)
    # 旋转中心
    center = (new_w / 2, new_h / 2)
    M_rot = cv2.getRotationMatrix2D(center, rotation, 1.0)

    # 转为 3x3 矩阵
    H = np.eye(3, dtype=np.float32)
    H[:2] = M_rot

    # 加入透视因子 (在投影矩阵的最后一行)
    H[2, 0] = px
    H[2, 1] = py

    # 应用 Homography
    # 这种方式可以产生梯形变换
    warped = cv2.warpPerspective(resized, H, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)

    # 4. 平移 (贴图)
    canvas = np.zeros((h_t, w_t), dtype=img.dtype)
    if img.ndim == 3:
        canvas = np.zeros((h_t, w_t, img.shape[2]), dtype=img.dtype)

    x_start = int(dx)
    y_start = int(dy)

    src_x1, src_y1 = 0, 0
    src_x2, src_y2 = new_w, new_h
    dst_x1, dst_y1 = x_start, y_start
    dst_x2, dst_y2 = x_start + new_w, y_start + new_h

    # 边界裁剪
    if dst_x1 < 0: src_x1 -= dst_x1; dst_x1 = 0
    if dst_y1 < 0: src_y1 -= dst_y1; dst_y1 = 0
    if dst_x2 > w_t: src_x2 -= (dst_x2 - w_t); dst_x2 = w_t
    if dst_y2 > h_t: src_y2 -= (dst_y2 - h_t); dst_y2 = h_t

    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = warped[src_y1:src_y2, src_x1:src_x2]

    return canvas
