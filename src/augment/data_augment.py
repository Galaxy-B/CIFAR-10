import cv2

def augment_image(img):
    augmented = []

    # 原图
    augmented.append(img)

    # 水平翻转
    augmented.append(cv2.flip(img, 1))

    # 轻微旋转
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    for angle in [-15, 15]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented.append(rotated)

    # 改变亮度
    brighter = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
    darker = cv2.convertScaleAbs(img, alpha=0.8, beta=-20)
    augmented.extend([brighter, darker])

    return augmented
