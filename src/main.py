import pandas as pd
from sklearn.preprocessing import StandardScaler

from load_data.load_image import load_image

from extract_feature.lbp import lbp
from extract_feature.gabor_filter import gabor_filter_features
from extract_feature.moments import moments
from extract_feature.hog import hog_features

from reduce_dimension.pca import pca

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report

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
    # feat_handles = [lbp, gabor_filter_features, moments]
    feat_handles = [hog_features]

    # train_features = pd.concat([handle(train_images) for handle in feat_handles], axis=1)
    # test_features = pd.concat([handle(test_images) for handle in feat_handles], axis=1)

    train_features = pd.concat([handle(train_images) for handle in feat_handles], axis=1).values
    test_features = pd.concat([handle(test_images) for handle in feat_handles], axis=1).values

    print(f"train features shape -> {train_features.shape}")
    print(f"test features shape -> {test_features.shape}")

    # 特征处理
    # train_features = pca(train_features.values, n_components=50)
    # test_features = pca(test_features.values, n_components=50)

    # scaler = StandardScaler()
    # train_features = scaler.fit_transform(train_features)
    # test_features = scaler.fit_transform(test_features)

    print(f"train features shape after PCA -> {train_features.shape}")
    print(f"test features shape after PCA -> {test_features.shape}")

    # 训练模型
    # model = KNeighborsClassifier(n_neighbors=5)
    # model.fit(train_features, train_labels)

    model = SVC(kernel='linear', C=1.0, random_state=42)
    model.fit(train_features, train_labels)

    # TODO: 评估模型
    yhat = model.predict(test_features)

    train_yhat = model.predict(train_features)

    acc = 0
    for i, j in zip(train_labels, train_yhat):
        acc += int(i == j)

    print("正确率: {:.2f}%".format(acc / len(train_labels) * 100))

    print("准确率: {:.2f}%".format(accuracy_score(test_labels, yhat) * 100))
    print("\n分类报告:")
    print(classification_report(test_labels, yhat))
