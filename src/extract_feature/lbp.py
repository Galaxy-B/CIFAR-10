import cv2
import numpy as np
import pandas as pd

from numpy import ndarray
from skimage.feature import local_binary_pattern

def calculate_lbp(image: ndarray):
    # 将原始图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算LBP特征及其直方图
    lbp: ndarray = local_binary_pattern(gray, P=4, R=2, method="uniform")
    max_bin = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=max_bin, range=(0, max_bin))

    # 归一化直方图
    hist = hist.astype(float)
    return hist / (hist.sum() + 1e-6)

"""
特征提取使用统一格式的接口, 以便直接嵌入框架:
输入 -> 图像列表: list[ndarray]
输出 -> 特征值DataFrame: pd.DataFrame
"""

def lbp(images: list) -> pd.DataFrame:
    # LBP提取结果是分箱的列表 需要展开为多个特征
    df = pd.DataFrame([calculate_lbp(image) for image in images])
    df.columns = [f"lbp_{i}" for i in range(len(df.columns))]
    return df
