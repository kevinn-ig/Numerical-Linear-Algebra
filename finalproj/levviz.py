import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_matrix

def cur_decomposition(A, k):
    # Compute SVD of A
    U, s, Vt = la.svd(A.toarray(), full_matrices=False)  # Convert sparse matrix to dense for SVD
    V = Vt.T

    # Compute raw leverage scores
    raw_leverage_scores = np.sum(V**2, axis=1)

    # Normalize the scores to form a probability distribution
    leverage_scores = raw_leverage_scores / np.sum(raw_leverage_scores)
    
    # Select k columns based on the leverage scores probabilistically
    selected_columns = np.random.choice(np.arange(A.shape[1]), size=k, replace=False, p=leverage_scores)
    
    C = A[:, selected_columns].toarray()
    R = A[selected_columns, :].toarray()
    C_pinv = la.pinv(C)
    R_pinv = la.pinv(R)
    U = C_pinv @ A.toarray() @ R_pinv
    
    return C, U, R, selected_columns, leverage_scores

# Create a sparse example matrix A
data = np.array([1, 2, 3, 4, 5, 2, 4, 6])
row_indices = np.array([0, 0, 0, 1, 1, 2, 2, 2])
col_indices = np.array([0, 2, 4, 1, 3, 0, 2, 4])
A = csr_matrix((data, (row_indices, col_indices)), shape=(5, 10))

# Number of columns to select
k = 

# Perform CUR decomposition
C, U, R, selected_columns, leverage_scores = cur_decomposition(A, k)

print("Matrix C:\n", C)
print("Matrix U:\n", U)
print("Matrix R:\n", R)
print("Selected Columns:", selected_columns)
print("Leverage Scores:", leverage_scores)
print("Matrix A:", A)
