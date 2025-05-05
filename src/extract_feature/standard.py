import numpy as np

# 对特征向量进行归一化处理
def normalize_features(features: np.ndarray, method: str='minmax'):
    if method == 'l2':
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 防止除零
        normalized_features = features / norms
    elif method == 'minmax':
        min_vals = np.min(features, axis=0, keepdims=True)
        max_vals = np.max(features, axis=0, keepdims=True)
        
        denom = max_vals - min_vals
        denom[denom == 0] = 1  # 防止除零
        normalized_features = (features - min_vals) / denom
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    return normalized_features
