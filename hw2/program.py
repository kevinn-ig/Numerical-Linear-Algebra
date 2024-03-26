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
    A = np.array(L)@np.array(U)

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
                result[i][j] = sum(L[i][k] * U[j][k] for k in range(n))
    return result


def factorization_accuracy(A, L, U, P, Q):
    PA = np.matmul(np.matmul(P,A), Q)
    LU = np.matmul(L, U)
    error = np.linalg.norm(PA - LU, ord='fro') / np.linalg.norm(A)
    return error

def growth_factor(L, U, A):
    LU_norm = np.linalg.norm(np.matmul(abs(L), abs(U)), ord='fro')
    A_norm = np.linalg.norm(A, ord='fro')
    return LU_norm /  A_norm

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

# Experiment parameters
problem_sizes = [5,10,50,100]
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
            error = factorization_accuracy(A, L, U, P, Q)
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
