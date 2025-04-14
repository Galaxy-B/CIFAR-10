import pandas as pd

from load_data.load_image import load_image
from extract_feature.lbp import lbp
from extract_feature.gabor_filter import gabor_filter_features
from extract_feature.moments import moments
from extract_feature.hog import hog_features

if __name__ == "__main__":
    # 读取图片
    train_images, train_labels = load_image("data\\train")
    test_images, test_labels = load_image("data\\test")

    print(f"train data -> images: {len(train_images)}, labels: {len(train_labels)}")
    print(f"test data -> images: {len(test_images)}, labels: {len(test_labels)}")

    # 提取特征
    train_features = pd.DataFrame()
    test_features = pd.DataFrame()

    # --- 将提取特征的接口注册在这里 ---
    # feat_handles = [lbp, gabor_filter_features, moments, hog_features]
    feat_handles = [lbp, gabor_filter_features, moments]

    train_features = pd.concat([handle(train_images) for handle in feat_handles], axis=1)
    test_features = pd.concat([handle(test_images) for handle in feat_handles], axis=1)

    print(f"train features shape -> {train_features.shape}")
    print(f"test features shape -> {test_features.shape}")

    # TODO: 特征处理

    # TODO: 训练模型

    # TODO: 评估模型
