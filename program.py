import random
import matplotlib.pyplot as plt
import time

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
def lu_factorization(A, pivoting):
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



def matrix_norm(W, norm_type):
    n = len(W)
    if norm_type == '1':
        return max(sum(abs(W[j][i]) for j in range(n)) for i in range(n))
    elif norm_type == 'inf':
        return max(sum(abs(W[i][j]) for j in range(n)) for i in range(n))
    elif norm_type == 'F':
        return sum(sum(W[i][j] ** 2 for j in range(n)) for i in range(n)) ** 0.5
    else:
        raise ValueError("Invalid norm type. Choose from '1', 'inf', or 'F'")

def factorization_accuracy(A, L, U, P):
    PA = apply_permutation_matrix(P, A)
    LU = matrix_multiplication(L, U, 'LU')
    error = matrix_norm(subtract_matrices(PA, LU), 'F') / max(1, matrix_norm(A, 'F'))
    return error

def growth_factor(L, U, A):
    LU_norm = matrix_norm(matrix_multiplication(L, U, 'LU'), 'F')
    A_norm = matrix_norm(A, 'F')
    return LU_norm / max(1, A_norm)

# Experiment parameters
problem_sizes = [10, 50, 100, 300, 500]
num_samples = 10
results = []

for i in problem_sizes:
    accuracies = []
    growth_factors = []
    execution_times = []

    for _ in range(num_samples):
        A = generate_full_rank_matrix(i)
        
        start_time = time.time()
        P, L, U = lu_factorization(A, pivoting='complete')
        end_time = time.time()

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
        'n': i,
        'avg_accuracy': avg_accuracy,
        'avg_growth_factor': avg_growth_factor,
        'avg_execution_time': avg_execution_time
    
    })
    print(f"n = {i} done")

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot([r['n'] for r in results], [r['avg_accuracy'] for r in results], marker='o')
plt.xlabel('Problem Size (n)')
plt.ylabel('Average Factorization Accuracy')
plt.title('Factorization Accuracy vs. Problem Size')

plt.subplot(1, 3, 2)
plt.plot([r['n'] for r in results], [r['avg_growth_factor'] for r in results], marker='o')
plt.xlabel('Problem Size (n)')
plt.ylabel('Average Growth Factor')
plt.title('Growth Factor vs. Problem Size')

plt.subplot(1, 3, 3)
plt.plot([r['n'] for r in results], [r['avg_execution_time'] for r in results], marker='o')
plt.xlabel('Problem Size (n)')
plt.ylabel('Average Execution Time (s)')
plt.title('Execution Time vs. Problem Size')

plt.tight_layout()
plt.show()