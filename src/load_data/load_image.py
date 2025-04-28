import os
import cv2

def equalize_image(img):
    """
    对单张图像做直方图均衡化。
    - 灰度图：直接 cv2.equalizeHist
    - 彩色图：转换到 YCrCb 空间，对亮度通道做均衡化，再转回 BGR
    """
    # 将图像转换为灰度图
    # return cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def load_image(path: str) -> tuple[list, list]:
    image_list = []
    label_list = []

    for entry in os.listdir(path):    
        full_path = os.path.join(path, entry)

        image_list.append(equalize_image(cv2.imread(full_path)))
        label_list.append(int(entry.split('_', 1)[0]))

    return image_list, label_list
