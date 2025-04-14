import cv2
import numpy as np
import pandas as pd

from numpy import ndarray
from skimage.feature import hog

def calculate_hog(image: ndarray):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算HOG特征（适用于32x32图像）
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(4, 4),     # 小cell以更细致捕捉局部结构
        cells_per_block=(2, 2),     # 每个block包含2x2个cell
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )
    return features

"""
统一格式接口:
输入 -> 图像列表: list[ndarray]
输出 -> 特征值DataFrame: pd.DataFrame
"""

def hog_features(images: list) -> pd.DataFrame:
    features = [calculate_hog(image) for image in images]
    df = pd.DataFrame(features)
    df.columns = [f"hog_{i}" for i in range(df.shape[1])]
    return df