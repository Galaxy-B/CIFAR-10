import cv2
import numpy as np
import pandas as pd

from extract_feature.standard import normalize_features
from numpy import ndarray

def calculate_moments(image: ndarray):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算图像的空间矩
    moments = cv2.moments(image)

    # 计算Hu矩（7个值），并对其进行log变换增强数值稳定性
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    return hu_moments

"""
统一格式接口:
输入 -> 图像列表: list[ndarray]
输出 -> 特征值DataFrame: pd.DataFrame
"""

def moments(images: list) -> pd.DataFrame:
    df = pd.DataFrame([normalize_features(calculate_moments(image)) for image in images])
    df.columns = [f"hu_{i}" for i in range(len(df.columns))]
    return df