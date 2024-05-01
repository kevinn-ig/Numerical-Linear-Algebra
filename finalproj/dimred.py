import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from numpy.linalg import pinv

def cur_decomposition(matrix, k):
    m, n = matrix.shape
    row_indices = np.random.choice(m, k, replace=False)
    col_indices = np.random.choice(n, k, replace=False)
    C = matrix[:, col_indices]
    R = matrix[row_indices, :]
    U = pinv(C) @ matrix @ pinv(R)
    approximation = C @ U @ R
    return C,U,R

# Generate synthetic high-dimensional data
X, y = make_classification(n_samples=100, n_features=50, n_informative=10, n_redundant=40, random_state=42)

# Apply CUR decomposition to reduce dimensions
C, U, R = cur_decomposition(X, 10)  # Selecting 10 dimensions
reduced_X = C @ U

# Visualization
plt.figure(figsize=(10, 5))
plt.scatter(reduced_X[:, 0], reduced_X[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('CUR Component 1')
plt.ylabel('CUR Component 2')
plt.title('Dimensionality Reduction with CUR')
plt.colorbar()

plt.show()
