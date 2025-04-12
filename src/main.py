import pandas as pd

from src.load_data.load_image import load_image
from src.extract_feature.lbp import lbp

if __name__ == "__main__":
    # 读取图片
    train_images, train_labels = load_image("data\\train")
    test_images, test_labels = load_image("data\\test")

    # 提取特征
    train_features = pd.DataFrame()
    test_features = pd.DataFrame()

    for feat_handle in [lbp]:
        train_features = pd.concat(train_features, feat_handle(train_images), axis=1)
        test_features = pd.concat(test_features, feat_handle(test_images), axis=1)

    # TODO: 特征处理

    # TODO: 训练模型

    # TODO: 评估模型
