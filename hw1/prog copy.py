# Try Identity Matrices.

import numpy as np
import random
import csv
import matplotlib.pyplot as plt

def U_gen(n):
    upper_triangular_matrix = {}
    for i in range(n):
        upper_triangular_matrix[(i, i)] = random.randint(-10, 10)
        while upper_triangular_matrix[(i, i)] == 0:
            upper_triangular_matrix[(i, i)] = random.randint(-10, 10)
        for j in range(i + 1, n):
            upper_triangular_matrix[(i, j)] = random.randint(0, 10)
    return upper_triangular_matrix

def L_gen(n):
    lower_triangular_matrix = {}
    for i in range(n):
        lower_triangular_matrix[(i, i)] = 1
        for j in range(i):
            lower_triangular_matrix[(i, j)] = random.randint(0, 10)
    return lower_triangular_matrix

def matrix_vector_multiply(M, v):
    result = {}
    for (i, j), value in M.items():
        if i not in result:
            result[i] = 0
        result[i] += value * v[j]
    return [result[i] for i in range(len(v))]

def matrix_matrix_multiply(L, U):
    result = {}
    for (i, k1), L_value in L.items():
        for (k2, j), U_value in U.items():
            if k1 == k2:
                if (i, j) not in result:
                    result[(i, j)] = 0
                result[(i, j)] += L_value * U_value
    return result

def dense_matrix(matrix_dict, n):
    dense = np.zeros((n, n))
    for (i, j), value in matrix_dict.items():
        dense[i][j] = value
    return dense

def solve_lower_triangular(L, b):
    n = len(b)
    y = np.zeros(n)
    for j in range(n):
        y[j] = b[j] / L[(j, j)]
        for i in range(j + 1, n):
            b[i] -= L[(i, j)] * y[j]
    return y

def solve_upper_triangular(U, y):
    n = len(y)
    x = np.zeros(n)
    for j in range(n - 1, -1, -1):
        x[j] = y[j] / U[(j, j)]
        for i in range(j - 1, -1, -1):
            y[i] -= U[(i, j)] * x[j]
    return x

abs_errors_y_all = []
rel_errors_y_all = []
abs_errors_x_all = []
rel_errors_x_all = []

n_values = list(range(10, 201, 10))

for n in n_values:
    nan_or_inf = True
    while nan_or_inf:
        v = [random.randint(-10, 10) for _ in range(n)]
        b = [random.randint(-10, 10) for _ in range(n)]
        L = L_gen(n)
        U = U_gen(n)
        y = solve_lower_triangular(L, b)
        x = solve_upper_triangular(U, y)

        # Check for NaN or Inf in the results
        if not (np.isnan(y).any() or np.isnan(x).any() or np.isinf(y).any() or np.isinf(x).any()):
            nan_or_inf = False

    # Convert custom matrices and vectors to NumPy arrays
    L_np = dense_matrix(L, n)
    U_np = dense_matrix(U, n)
    v_np = np.array(v)
    b_np = np.array(b)
    y_np = np.array(y)
    x_np = np.array(x)

    # Calculate absolute and relative errors for the solvers using the two-norms
    vec_sub = np.linalg.solve(L_np, b_np) - y_np
    abs_error_y = np.linalg.norm(vec_sub)
    rel_error_y = abs_error_y / np.linalg.norm(y_np)

    vec_sub = np.linalg.solve(U_np, y_np) - x_np
    abs_error_x = np.linalg.norm(vec_sub)
    rel_error_x = abs_error_x / np.linalg.norm(x_np)

    # Store results in CSV
    with open(f'errors_n_{n}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Abs_Error_y', 'Rel_Error_y', 'Abs_Error_x', 'Rel_Error_x'])
        writer.writerow([abs_error_y, rel_error_y, abs_error_x, rel_error_x])

    # Append mean errors to the lists
    abs_errors_y_all.append(abs_error_y)
    rel_errors_y_all.append(rel_error_y)
    abs_errors_x_all.append(abs_error_x)
    rel_errors_x_all.append(rel_error_x)

# Calculate mean relative errors over all n
mean_rel_error_y_all = np.mean(rel_errors_y_all)
mean_rel_error_x_all = np.mean(rel_errors_x_all)
mean_abs_error_y_all = np.mean(abs_errors_y_all)
mean_abs_error_x_all = np.mean(abs_errors_x_all)

print(f"Overall Mean Relative Error Solver y: {mean_rel_error_y_all}")
print(f"Overall Mean Relative Error Solver x: {mean_rel_error_x_all}")
print(f"Overall Mean Absolute Error Solver y: {mean_abs_error_y_all}")
print(f"Overall Mean Absolute Error Solver x: {mean_abs_error_x_all}")

# Create line graphs for errors over n
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(n_values, rel_errors_y_all, marker='o', color='green')
plt.title('Mean Relative Error Solver y over n')
plt.xlabel('n')
plt.ylabel('Mean Relative Error')

plt.subplot(1, 2, 2)
plt.plot(n_values, rel_errors_x_all, marker='o', color='red')
plt.title('Mean Relative Error Solver x over n')
plt.xlabel('n')
plt.ylabel('Mean Relative Error')

plt.tight_layout()
plt.savefig("means.png")

# Create histograms
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(rel_errors_y_all, bins=20, color='blue', edgecolor='black')
plt.title(f'Relative Error Solver y ')

plt.subplot(1, 2, 2)
plt.hist(rel_errors_x_all, bins=20, color='green', edgecolor='black')
plt.title(f'Relative Error Solver x')
plt.savefig("histograms.png")
