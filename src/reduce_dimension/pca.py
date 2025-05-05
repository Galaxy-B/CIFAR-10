import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca(features: np.ndarray, n_components: int = 10) -> np.ndarray:
    # 归一化（标准化）
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # PCA降维到10维
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)

    return features_pca
