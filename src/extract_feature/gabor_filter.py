import cv2
import numpy as np
import pandas as pd

def gabor_filter_features(images: list) -> pd.DataFrame:
    """
    使用Gabor滤波器提取图像特征并返回DataFrame。
    
    参数:
    images (list[ndarray]): 输入图像列表,ndarray格式的图像。
    
    返回:
    pd.DataFrame: 提取的特征值DataFrame,每行对应一张图像的特征向量。
    """
    # Gabor滤波器的参数设置
    kernels = []
    for theta in np.arange(0, np.pi, np.pi / 4):  # 设置不同方向的Gabor核
        for lamda in [np.pi / 4, np.pi / 2]:  # 设置不同的波长
            kernel = cv2.getGaborKernel((21, 21), 5.0, theta, lamda, 1.0, 0, ktype=cv2.CV_32F)
            kernels.append(kernel)
    
    # 存储所有图像的特征
    features = []

    for img in images:
        # 转换为灰度图像
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计算每个Gabor滤波器的响应
        responses = []
        for kernel in kernels:
            filtered_img = cv2.filter2D(gray_img, cv2.CV_32F, kernel)
            responses.append(np.mean(filtered_img))  # 使用平均值作为特征
        
        features.append(responses)

    # 转换为DataFrame
    feature_df = pd.DataFrame(features)
    return feature_df
