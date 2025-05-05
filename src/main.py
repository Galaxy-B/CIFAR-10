import pandas as pd
from load_data.load_image import load_image
from augment.data_augment import augment_image

from extract_feature.lbp import lbp
from extract_feature.gabor_filter import gabor_filter_features
from extract_feature.moments import moments
from extract_feature.hog import hog_features

from sklearn.preprocessing import StandardScaler
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

    # 对训练集做数据增强
    augmented_images = []
    augmented_labels = []

    for img, label in zip(train_images, train_labels):
        augmented = augment_image(img)
        augmented_images.extend(augmented)
        augmented_labels.extend([label] * len(augmented))

    train_images = augmented_images
    train_labels = augmented_labels

    print(f"augmented train data -> images: {len(train_images)}, labels: {len(train_labels)}")

    # 提取特征
    train_features = pd.DataFrame()
    test_features = pd.DataFrame()

    # --- 将提取特征的接口注册在这里 ---
    # feat_handles = [lbp, gabor_filter_features, moments, hog_features]
    # feat_handles = [lbp, gabor_filter_features, moments]
    feat_handles = [hog_features]

    train_features = pd.concat([handle(train_images) for handle in feat_handles], axis=1)
    test_features = pd.concat([handle(test_images) for handle in feat_handles], axis=1)

    train_features.head(10).to_csv("features.csv")  # 便于查看数据特点

    print(f"\ntrain features shape -> {train_features.shape}")
    print(f"test features shape -> {test_features.shape}")

    # 特征处理
    # train_features = pca(train_features.values, n_components=50)
    # test_features = pca(test_features.values, n_components=50)

    # scaler = StandardScaler()
    # train_features = scaler.fit_transform(train_features)
    # test_features = scaler.fit_transform(test_features)

    train_features = train_features.values
    test_features = test_features.values

    # print(f"train features shape after PCA -> {train_features.shape}")
    # print(f"test features shape after PCA -> {test_features.shape}")

    # 训练模型
    # model = KNeighborsClassifier(n_neighbors=5)
    model = SVC(kernel='linear', C=0.5, random_state=42)
    model.fit(train_features, train_labels)

    # 评估模型
    train_yhat = model.predict(train_features)
    test_yhat = model.predict(test_features)

    print("\nACC on train data: {:.2f}%".format(accuracy_score(train_labels, train_yhat) * 100))
    print("ACC on test data: {:.2f}%".format(accuracy_score(test_labels, test_yhat) * 100))

    print("\n分类报告:")
    print(classification_report(test_labels, test_yhat))
