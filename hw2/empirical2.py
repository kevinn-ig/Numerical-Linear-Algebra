import numpy as np


def generate_antidiagonal_matrix(n):
    A = np.zeros((n, n))
    for i in range(n):
        A[i][n - i - 1] = n - i
    return A


def lu_decomposition_no_pivot(A):
    n = len(A)
    P = np.eye(n)
    Q = np.eye(n)
    L = np.zeros((n, n))
    U = np.array(A.copy())
    A = np.array(A)


    for k in range(n):
        L[k][k] = 1.0
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]
            U[i][k] = 0.0
            for j in range(k + 1, n):
                U[i][j] -= L[i][k] * U[k][j]


    return P, Q, L, U

def lu_decomposition_partial_pivot(A):
    n = len(A)
    P = np.eye(n)
    Q = np.eye(n)
    L = np.zeros((n, n))
    U = np.array(A.copy())
    A = np.array(A)

    for k in range(n): #loops through columns
        pivot_row = max(range(k, n), key=lambda i: abs(U[i][k]))
        if pivot_row != k:
            U[[k, pivot_row]] = U[[pivot_row, k]]
            P[[k, pivot_row]] = P[[pivot_row, k]]
        L[k][k] = 1.0
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]
            U[i][k] = 0.0
            for j in range(k + 1, n):
                U[i][j] -= L[i][k] * U[k][j]
    return P, Q, L, U

def lu_decomposition_complete_pivot(A):
    n = len(A)
    P = np.eye(n)
    Q = np.eye(n)
    L = np.zeros((n, n))
    U = np.array(A.copy())
    A = np.array(A)

    for k in range(n):
        pivot_element = max((abs(U[i][j]), i, j) for i in range(k, n) for j in range(k, n))
        i, j = pivot_element[1], pivot_element[2]
        if i != k:
            P[[k, i]] = P[[i, k]]
            U[[k, i]] = U[[i,k]] 
        if j != k:
            Q[:,[k, j]] = Q[:,[j, k]]
            U[:,[k, j]] = U[:,[j, k]]
        L[k][k] = 1.0
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]
            U[i][k] = 0.0
            for j in range(k + 1, n):
                U[i][j] -= L[i][k] * U[k][j]

    return P, Q, L, U

def perform_lu_decomposition(A, pivot_type):
    if pivot_type == "partial":
        P, Q, L, U = lu_decomposition_partial_pivot(A)
    elif pivot_type == "complete":
        P, Q, L, U = lu_decomposition_complete_pivot(A)
    elif pivot_type == "none":
        P, Q, L, U = lu_decomposition_no_pivot(A)
    else:
        raise ValueError("Invalid pivot_type. Use 'partial', 'complete', or 'none'.")
    return P, Q, L, U


# User input for pivot type
pivot_types = ["none", "partial", "complete"]
n = 5
A = generate_antidiagonal_matrix(n)

for i in range(len(pivot_types)):
    # Test the function
    print("Matrix A:")
    print(A)
    P, Q, L, U = perform_lu_decomposition(A, pivot_types[i])

    print(f"\nPivot_type = {pivot_types[i]}")
    print("\nMatrix L:")
    print(L)

    print("\nMatrix U:")
    print(U)

    # Check the factorization
    print("\nCheck LU - A:")
    print(L @ U - P@A@Q)