import numpy as np

def vector_norm_3d(v):
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        return float(np.sqrt(np.sum(v**2)))
    else:
        return np.sqrt(np.sum(v**2, axis=1))

# Test cases
print(vector_norm_3d([3, 4, 12]))              # → 13.0
print(vector_norm_3d([[1, 0, 0], [0, 3, 4]])) # → [1.0, 5.0]