import random
import matplotlib.pyplot as plt
import time
import numpy as np
# Add so it calculates everything for complete, partial, and no pivoting


def generate_full_rank_matrix(n):
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    # Generate L as a unit lower trapezoidal matrix
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                L[i][j] = 1.0
            else:
                L[i][j] = random.uniform(0.1, 1.0)  # Ensure off-diagonal elements are not too small

    # Generate U as an upper triangular matrix
    for i in range(n):
        for j in range(i, n):
            if i == j:
                U[i][j] = random.uniform(1.0, 10.0)  # Ensure diagonal elements are not too small
            else:
                U[i][j] = random.uniform(0.1, 1.0)  # Ensure off-diagonal elements are not too small

    # Compute A = LU
    A = [[sum(L[i][k] * U[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

    return A

def apply_permutation_matrix(P, A):
    n = len(A)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = A[P[i]][j]
    return result

def subtract_matrices(A, B):
    n = len(A)
    result = [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]
    return result

def matrix_multiplication(L, U, product_type):
    n = len(L)
    result = [[0.0] * n for _ in range(n)]
    if product_type == 'LU':
        for i in range(n):
            for j in range(n):
                result[i][j] = sum(L[i][k] * U[k][j] for k in range(n))
    elif product_type == '|LU|':
        for i in range(n):
            for j in range(n):
                result[i][j] = L[i][j] * U[j][i]
    return result

# LU factorization with pivoting
#def lu_factorization(A, pivoting):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    P = list(range(n))  # Initialize permutation matrix as identity permutation

    for i in range(n):
        if pivoting == 'partial':
            pivot_index = max(range(i, n), key=lambda x: abs(A[x][i]))
            if A[pivot_index][i] == 0:
                print("Partial pivoting failed. Zero pivot encountered.")
                return None, None, None
            A[i], A[pivot_index] = A[pivot_index], A[i]
            P[i], P[pivot_index] = P[pivot_index], P[i]
        elif pivoting == 'complete':
            max_index = max(((k, j) for k in range(i, n) for j in range(i, n)), key=lambda x: abs(A[x[0]][x[1]]))
            if A[max_index[0]][max_index[1]] == 0:
                print("Complete pivoting failed. Zero pivot encountered.")
                return None, None, None
            A[i], A[max_index[0]] = A[max_index[0]], A[i]
            for k in range(n):
                A[k][i], A[k][max_index[1]] = A[k][max_index[1]], A[k][i]
            P[i], P[max_index[0]] = P[max_index[0]], P[i]

        L[i][i] = 1.0
        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i+1, n):
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    return P, L, U


def factorization_accuracy(A, L, U, P):
    PA = np.matmul(P, A)
    LU = np.matmul(L, U)
    error = np.linalg.norm(PA - LU, ord='fro') / max(1, np.linalg.norm(A, ord='fro'))
    return error

def growth_factor(L, U, A):
    LU_norm = np.linalg.norm(np.matmul(L, U), ord='fro')
    A_norm = np.linalg.norm(A, ord='fro')
    return LU_norm / max(1, A_norm)

def lu_decomposition_no_pivot(A):
    n = len(A)
    P = list(range(n))
    Q = list(range(n))
    L = [[0.0] * n for _ in range(n)]
    U = [row.copy() for row in A]

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
    P = [i for i in range(n)]
    Q = [i for i in range(n)]
    L = [[0.0] * n for _ in range(n)]
    U = [row.copy() for row in A]

    for k in range(n):
        pivot_row = max(range(k, n), key=lambda i: abs(U[i][k]))
        if pivot_row != k:
            A[k], A[pivot_row] = A[pivot_row], A[k]
            P[k], P[pivot_row] = P[pivot_row], P[k]
        L[k][k] = 1.0
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]
            U[i][k] = 0.0
            for j in range(k + 1, n):
                U[i][j] -= L[i][k] * U[k][j]
        for i in range(k + 1, n):
            U[i][k] = 0

    return P, Q, L, U

def lu_decomposition_complete_pivot(A):
    n = len(A)
    P = [i for i in range(n)]
    Q = [i for i in range(n)]
    L = [[0.0] * n for _ in range(n)]
    U = [row.copy() for row in A]

    for k in range(n):
        max_val = 0
        max_index = (0, 0)
        for i in range(k, n):
            for j in range(k, n):
                if abs(U[i][j]) > max_val:
                    max_val = abs(U[i][j])
                    max_index = (i, j)
        i, j = max_index
        if i != k:
            A[k], A[i] = A[i], A[k]
            P[k], P[i] = P[i], P[k]
        if j != k:
            A = [list(row) for row in zip(*A)]  # Transpose A
            A[k], A[j] = A[j], A[k]
            A = [list(row) for row in zip(*A)]  # Transpose back
            Q[k], Q[j] = Q[j], Q[k]
        L[k][k] = 1.0
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]
            U[i][k] = 0.0
            for j in range(k + 1, n):
                U[i][j] -= L[i][k] * U[k][j]
        for i in range(k + 1, n):
            U[i][k] = 0

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

# Experiment parameters
problem_sizes = [10, 20, 30]
num_samples = 10
results = []
pivot_types = ["none", "partial", "complete"]

for pivot_type in pivot_types:
    for j in range(len(problem_sizes)):
        accuracies = []
        growth_factors = []
        execution_times = []

        for _ in range(num_samples):
            A = generate_full_rank_matrix(problem_sizes[j])
            numpy_A = np.array(A)

            start_time = time.time()
            P, Q, L, U = perform_lu_decomposition(A, pivot_type)
            end_time = time.time()
            numpy_L = np.array(L)
            numpy_U = np.array(U)

            # Calculate factorization accuracy and growth factor
            error = factorization_accuracy(A, L, U, P)
            gamma = growth_factor(L, U, A)

            # Store results
            accuracies.append(error)
            growth_factors.append(gamma)
            execution_times.append(end_time - start_time)

        # Calculate average metrics
        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_growth_factor = sum(growth_factors) / len(growth_factors)
        avg_execution_time = sum(execution_times) / len(execution_times)

        results.append({
            'pivot_type': pivot_type,
            'n': problem_sizes[j],
            'avg_accuracy': avg_accuracy,
            'avg_growth_factor': avg_growth_factor,
            'avg_execution_time': avg_execution_time
        })

        print(f"Pivot type = {pivot_type}, n = {problem_sizes[j]} done")

# Plot results
plt.figure(figsize=(12, 6))

for pivot_type in pivot_types:
    pivot_results = [r for r in results if r['pivot_type'] == pivot_type]

    plt.plot([r['n'] for r in pivot_results], [r['avg_accuracy'] for r in pivot_results], marker='o', label=pivot_type)

plt.xlabel('Problem Size (n)')
plt.ylabel('Average Factorization Accuracy')
plt.title('Factorization Accuracy vs. Problem Size')
plt.legend()
plt.savefig("acc_over_n.png")
plt.close()

plt.figure(figsize=(12, 6))

for pivot_type in pivot_types:
    pivot_results = [r for r in results if r['pivot_type'] == pivot_type]

    plt.plot([r['n'] for r in pivot_results], [r['avg_growth_factor'] for r in pivot_results], marker='o', label=pivot_type)

plt.xlabel('Problem Size (n)')
plt.ylabel('Average Growth Factor')
plt.title('Growth Factor vs. Problem Size')
plt.legend()
plt.savefig("grow_fac.png")
plt.close()

plt.figure(figsize=(12, 6))

for pivot_type in pivot_types:
    pivot_results = [r for r in results if r['pivot_type'] == pivot_type]

    plt.plot([r['n'] for r in pivot_results], [r['avg_execution_time'] for r in pivot_results], marker='o', label=pivot_type)

plt.xlabel('Problem Size (n)')
plt.ylabel('Average Execution Time (s)')
plt.title('Execution Time vs. Problem Size')
plt.legend()
plt.savefig("ex_time.png")
plt.close()
