import os
import cv2

def equalize_image(img):
    # 将图像转换为灰度图后做直方图均衡化
    # return cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    
    # 将图像转换为灰度图
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
