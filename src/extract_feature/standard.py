import numpy as np

def normalize_features(features, method='minmax'):
    """
    对特征向量进行归一化处理。

    Args:
        features (np.ndarray): 输入特征矩阵 (N_samples, N_features)
        method (str): 归一化方法，'l2' 或 'minmax'

    Returns:
        np.ndarray: 归一化后的特征矩阵
    """
    if method == 'l2':
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 防止除零
        normalized_features = features / norms
    elif method == 'minmax':
        min_vals = np.min(features, axis=0, keepdims=True)
        max_vals = np.max(features, axis=0, keepdims=True)
        # 防止除零
        denom = max_vals - min_vals
        denom[denom == 0] = 1
        normalized_features = (features - min_vals) / denom
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    return normalized_features