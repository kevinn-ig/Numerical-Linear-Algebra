import numpy as np
import matplotlib.pyplot as plt
import os

def generate_spd_matrix(n):
    """Generate a symmetric positive definite matrix of size n x n."""
    A = np.random.rand(n, n)
    A = A + A.T  # Make the matrix symmetric
    A += n * np.eye(n)  # Make the matrix positive definite
    return A

def plot_relative_error(method, scenarios, max_size, step=50):
    """Plot the relative error for the specified method and scenarios."""
    plt.figure(figsize=(12, 6))

    for scenario in scenarios:
        sizes = range(step, max_size + 1, step)
        errors = []


        for size in sizes:
            if scenario == "SPD":  # Square nonsingular matrix A
                A = generate_spd_matrix(size)
                x_true = np.random.rand(size)
                b = A @ x_true
            elif scenario == "Consistent System":  # Rectangular matrix A with full column rank, consistent system
                fullrankstatus = False
                while(fullrankstatus == False):
                    A = np.random.rand(size, size-2)
                    rank = np.linalg.matrix_rank(A)
                    if rank == size - 2:
                        fullrankstatus = True
                x_true = np.random.rand(size - 2)
                b = A.dot(x_true)
            elif scenario == "Inconsistent System":  # Rectangular matrix A with full column rank, inconsistent system
                fullrankstatus = False
                while not fullrankstatus:
                    A = np.random.rand(size, size - 2)
                    rank = np.linalg.matrix_rank(A)
                    if rank == size - 2:
                        fullrankstatus = True
                x_true = np.random.rand(size - 2)  # Least squares solution
                b1 = A @ x_true
                random_vec = np.random.rand(size)
                Q, _ = np.linalg.qr(A)
                projection = Q @ Q.T @ random_vec
                b2 = random_vec - projection
                if np.linalg.norm(b2) < 1e-6:
                    b2 += 1
                b = b1 + b2  # Inconsistent system

            else:
                raise ValueError("Invalid scenario number.")

            if method == 'least_squares':
                x_ls = least_squares(A, b)
                relative_error = np.linalg.norm(x_ls - x_true) / np.linalg.norm(x_true)
            elif method == 'incremental_least_squares':
                x_prev = np.zeros(A.shape[1])
                for i in range(1, size + 1):
                    A_inc = A[:i, :i]
                    b_inc = b[:i]
                    x_prev = incremental_least_squares(A_inc, b_inc, x_prev)
                relative_error = np.linalg.norm(x_prev - x_true) / np.linalg.norm(x_true)
            else:
                raise ValueError("Method must be 'least_squares' or 'incremental_least_squares'")

            errors.append(relative_error)

        plt.plot(sizes, errors, marker='o', label=f'{scenario}')

    plt.title(f'Relative Error vs. Size (Method: {method})')
    plt.xlabel('Size of A (n)')
    plt.ylabel('Relative Error')
    plt.legend()
    plt.tight_layout()
    filename = os.path.join(os.path.dirname(__file__), f"errorplot_{method}.png")
    plt.savefig(filename)  # Save the plot with the specified filename

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

size = 10


#1. n = k, i.e., a square nonsingular matrix A where xmin = A−1b.
#2. n > k and Ax = b for b ∈ Rn and b ∈ R(A) i.e., a rectangular matrix A with full
#column rank and a vector b that deﬁne a consistent set of overdetermined equations.
#3. n > k and b ∈ Rn and b 6 ∈ R(A) i.e., a rectangular matrix A with full column rank and
#a vector b = b1 + b2, b1 ∈ R(A), b2 6 ∈ R(A), b2 6 = 0, that deﬁne a linear least squares
#problem with a nonzero residual rmin = b2 = b − Axmin
#3

test_least_squares()
test_incremental_least_squares()

# Usage for scenarios 1, 2, and 3
plot_relative_error('least_squares', ["SPD", "Consistent System", "Inconsistent System"], max_size=200)
plot_relative_error('incremental_least_squares', ["SPD", "Consistent System", "Inconsistent System"], max_size=200)

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
