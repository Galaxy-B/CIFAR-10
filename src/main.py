import pandas as pd

from load_data.load_image import load_image
from extract_feature.lbp import lbp
from extract_feature.gabor_filter import gabor_filter_features

if __name__ == "__main__":
    # 读取图片
    train_images, train_labels = load_image("data\\train")
    test_images, test_labels = load_image("data\\test")

    # 提取特征
    train_features = pd.DataFrame()
    test_features = pd.DataFrame()

    # for feat_handle in [lbp]:
    #     train_features = pd.concat(train_features, feat_handle(train_images), axis=1)
    #     test_features = pd.concat(test_features, feat_handle(test_images), axis=1)

    train_features_gabor = gabor_filter_features(train_images)
    test_features_gabor = gabor_filter_features(test_images)
    print(train_features_gabor.shape, train_features_gabor)
    # TODO: 特征处理

    # TODO: 训练模型

    # TODO: 评估模型
