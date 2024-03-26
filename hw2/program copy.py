import random
import matplotlib.pyplot as plt
import time
import numpy as np

def generate_full_rank_matrix(n):
    L = np.tril(np.random.uniform(0.1, 1.0, (n, n)))
    U = np.triu(np.random.uniform(0.1, 1.0, (n, n))) + np.random.uniform(1.0, 10.0, (n, n))
    A = L.dot(U)
    return A

def apply_permutation_matrix(P, A):
    return A[P, :]

def subtract_matrices(A, B):
    return A - B

def matrix_multiplication(L, U, product_type):
    if product_type == 'LU':
        return np.matmul(L, U)
    elif product_type == '|LU|':
        return L * U.T


def factorization_accuracy(A, L, U, P, Q):
    PAQ = A
    LU = np.matmul(L, U)
    error = np.linalg.norm(PAQ - LU) / np.linalg.norm(PAQ)
    return error

def growth_factor(L, U, A):
    LU_norm = np.linalg.norm(np.matmul(L, U), ord='fro')
    A_norm = np.linalg.norm(A, ord='fro')
    return LU_norm / max(1, A_norm)

def lu_decomposition_no_pivot(A):
    n = len(A)
    P = list(range(n))
    Q = list(range(n))
    L = np.zeros((n, n))
    U = A.copy()


    for k in range(n):
        L[k][k] = 1.0
        for i in range(k + 1, n):
            L[i][k] = A[i][k] / A[k][k]
            U[i][k] = 0.0
            for j in range(k + 1, n):
                U[i][j] -= L[i][k] * A[k][j]


    return P, Q, L, U

def lu_decomposition_partial_pivot(A):
    n = len(A)
    P = np.eye(n)
    Q = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()

    for k in range(n): #loops through columns
        pivot_row = max(range(k, n), key=lambda i: abs(A[i][k]))
        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]
        L[k][k] = 1.0
        for i in range(k + 1, n):
            L[i][k] = A[i][k] / A[k][k]
            U[i][k] = 0.0
            for j in range(k + 1, n):
                U[i][j] -= L[i][k] * A[k][j]
        U[k+1:, k] = 0  #
    return P, Q, L, U

def lu_decomposition_complete_pivot(A):
    n = len(A)
    P = np.eye(n)
    Q = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()

    for k in range(n):
        pivot_element = max((abs(A[i][j]), i, j) for i in range(k, n) for j in range(k, n))
        i, j = pivot_element[1], pivot_element[2]
        if i != k:
            P[[k, i]] = P[[i, k]]
        if j != k:
            Q[:,[k, j]] = Q[:,[j, k]]
        L[k][k] = 1.0
        for i in range(k + 1, n):
            L[i][k] = A[i][k] / A[k][k]
            U[i][k] = 0.0
            for j in range(k + 1, n):
                U[i][j] -= L[i][k] * A[k][j]
    return P, Q, L, U

def perform_lu_decomposition(A, pivot_type):
    if pivot_type == "partial":
        P, Q, L, U = lu_decomposition_partial_pivot(A)
    elif pivot_type == "complete":
        P, Q, L, U = lu_decomposition_complete_pivot(A)
    elif pivot_type == "none":
        P,Q, L, U = lu_decomposition_no_pivot(A)
    else:
        raise ValueError("Invalid pivot_type. Use 'partial', 'complete', or 'none'.")

    return P, Q, L, U

problem_size = 10
pivot_type = 'partial'

# Generate a full rank matrix
A = generate_full_rank_matrix(problem_size)

# Perform LU decomposition
P, Q, L, U = perform_lu_decomposition(A, pivot_type)
print(P,Q)

# Reconstruct the original matrix
PAQ = A

# Calculate factorization accuracy
error = factorization_accuracy(A, L, U, P, Q)

# Display results
print("Original Matrix (A):\n", A)
print("\nReconstructed Matrix (PAQ):\n", PAQ)
print("\nFactorization Accuracy:", error)