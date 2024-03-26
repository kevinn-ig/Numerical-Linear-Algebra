import numpy as np
import matplotlib.pyplot as plt
import os

def householder_reflector(a):
    """Compute the Householder reflector for a vector a."""
    v = a.astype(float).copy()
    alpha = -np.sign(a[0]) * np.linalg.norm(a)
    v[0] -= alpha
    v /= np.linalg.norm(v)
    return np.eye(len(a)) - 2 * np.outer(v, v)


def qr_decomposition(A, tol=1e-12):
    """Compute the QR decomposition of matrix A using Householder reflectors."""
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    for i in range(n):
        H = np.eye(m)
        H[i:, i:] = householder_reflector(R[i:, i])
        Q = Q @ H
        R = H @ R
    # Set values below the tolerance to zero
    R[np.abs(R) < tol] = 0
    return Q, R

def least_squares(A, b):
    """Solve the linear least squares problem using QR decomposition."""
    Q, R = qr_decomposition(A)
    c = Q.T @ b
    x = np.linalg.solve(R[:A.shape[1]], c[:A.shape[1]])
    return x

def incremental_least_squares(A, b, x_prev=None):
    """Incremental update of the least squares solution."""
    if x_prev is None:
        x_prev = np.zeros(A.shape[1])
    A_new = np.vstack([A, np.zeros(A.shape[1])])
    b_new = np.append(b, 0)
    x_new = least_squares(A_new, b_new)
    return x_new

def create_laplacian_matrix(n):
    """Create the Laplacian matrix L for regularization."""
    L = -np.eye(n - 1, n) + np.eye(n - 1, n, k=1)
    return L

def regularized_least_squares(A, b, L, lambda_reg):
    """Solve the regularized linear least squares problem."""
    A_reg = np.vstack([A, np.sqrt(lambda_reg) * L])
    b_reg = np.append(b, np.zeros(L.shape[0]))
    x_reg = least_squares(A_reg, b_reg)
    return x_reg

def generate_signals(n, noise_variance=1.0):
    """Generate the true signal and the observed signal with noise."""
    t = np.linspace(0, 4, n)
    x_true = np.sin(t) + t * np.cos(t)**2
    noise = np.random.normal(0, np.sqrt(noise_variance), n)
    b = x_true + noise
    return t, x_true, b

def analyze_reconstruction(n_values, lambda_values, noise_variance=1.0):
    """Analyze the quality of the reconstruction for different n and lambda."""
    for n in n_values:
        t, x_true, b = generate_signals(n, noise_variance)
        plt.figure(figsize=(12, 8))
        for lambda_reg in lambda_values:
            L = create_laplacian_matrix(n)
            A = np.eye(n)  # Identity matrix for direct signal reconstruction
            x_reg = regularized_least_squares(A, b, L, lambda_reg)
            plt.plot(t, x_reg, label=f'λ={lambda_reg}')
        plt.plot(t, x_true, label='True signal', color='black', linewidth=2)
        plt.title(f'Signal Reconstruction for n={n}')
        plt.xlabel('Time')
        plt.ylabel('Signal')
        plt.legend()
        filename = os.path.join(os.path.dirname(__file__), f"signalplot_{n}.png")
        plt.savefig(filename)  # Save the plot with the specified filename

def test_least_squares():
    # Initialize a variable to track the number of failed tests
    failed_tests = 0

    # Test case 1: Known solution
    A = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([7, 8, 9])
    expected_solution = np.array([1.5, 2.0])  # Known solution for this test case
    computed_solution = least_squares(A, b)
    relative_error = np.linalg.norm(computed_solution - expected_solution) / np.linalg.norm(expected_solution)
    try:
        assert relative_error < 1e-12, f"Test case 1 failed with relative error {relative_error}"
    except AssertionError as e:
        print(e)
        failed_tests += 1

    # Test case 2: Comparison with numpy.linalg.lstsq
    A = np.random.rand(5, 3)
    b = np.random.rand(5)
    expected_solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    computed_solution = least_squares(A, b)
    relative_error = np.linalg.norm(computed_solution - expected_solution) / np.linalg.norm(expected_solution)
    try:
        assert relative_error < 1e-12, f"Test case 2 failed with relative error {relative_error}"
    except AssertionError as e:
        print(e)
        failed_tests += 1

    # Additional test cases with different dimensions and properties
    test_matrices = [
        (np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([9, 10, 11, 12])),
        (np.eye(4), np.arange(1, 5)),
        (np.array([[1, 1], [1, -1]]), np.array([2, 0]))
    ]
    for i, (A, b) in enumerate(test_matrices, start=3):
        expected_solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        computed_solution = least_squares(A, b)
        relative_error = np.linalg.norm(computed_solution - expected_solution) / np.linalg.norm(expected_solution)
        try:
            assert relative_error < 1e-12, f"Test case {i} failed with relative error {relative_error}"
        except AssertionError as e:
            print(e)
            failed_tests += 1

    if failed_tests == 0:
        print("All least squares test cases passed!")
    else:
        print(f"{failed_tests} least squares test case(s) failed.")

def test_incremental_least_squares():
    # Initialize a variable to track the number of failed tests
    failed_tests = 0

    # Test case: Incremental update
    A = np.array([[1, 2], [3, 4]])
    b = np.array([7, 8])
    x_prev = least_squares(A, b)
    A_new = np.vstack([A, [5, 6]])
    b_new = np.append(b, 9)
    expected_solution = least_squares(A_new, b_new)
    computed_solution = incremental_least_squares(A, b, x_prev)
    relative_error = np.linalg.norm(computed_solution - expected_solution) / np.linalg.norm(expected_solution)
    try:
        assert relative_error < 1e-12, f"Incremental test case failed with relative error {relative_error}"
    except AssertionError as e:
        print(e)
        failed_tests += 1

    if failed_tests == 0:
        print("All incremental least squares test cases passed!")
    else:
        print(f"{failed_tests} incremental least squares test case(s) failed.")


def test_regularized_least_squares():
    # Initialize a variable to track the number of failed tests
    failed_tests = 0

    # Test case: Regularized solution
    n = 10
    lambda_reg = 10
    L = create_laplacian_matrix(n)
    A = np.eye(n)
    b = np.random.rand(n)
    
    # Compute the regularized solution using your implementation
    computed_solution = regularized_least_squares(A, b, L, lambda_reg)
    
    # Construct the augmented system for the regularized problem
    A_aug = np.vstack([A, np.sqrt(lambda_reg) * L])
    b_aug = np.append(b, np.zeros(L.shape[0]))
    
    # Solve the augmented system using numpy.linalg.lstsq
    expected_solution, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    
    # Compare the computed solution with the expected solution
    if not np.allclose(computed_solution, expected_solution):
        print("Regularized test case failed")
        failed_tests += 1

    if failed_tests == 0:
        print("All regularized_least_squares test cases passed!")
    else:
        print(f"{failed_tests} regularized_least_squares test case(s) failed")

# Example usage
A = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([7, 8, 9])
x = least_squares(A, b)
print("Solution:", x)

test_least_squares()

A = np.array([[1, 2], [3, 4]])
b = np.array([7, 8])
x = incremental_least_squares(A, b)
print("Solution:", x)

test_incremental_least_squares()

# Example usage
n = 10
lambda_reg = 10
L = create_laplacian_matrix(n)  # This should create a matrix of size (n-1) x n
A = np.eye(n)  # Identity matrix of size n x n
b = np.random.rand(n)
x_reg = regularized_least_squares(A, b, L, lambda_reg)
print("Regularized solution:", x_reg)

test_regularized_least_squares()

# Example usage
n_values = [100, 200, 300, 400, 500]
lambda_values = [1, 10, 100, 1000]
analyze_reconstruction(n_values, lambda_values)

