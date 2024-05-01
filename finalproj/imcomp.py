import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.color import rgb2gray
from scipy.linalg import svd, pinv, diagsvd
from time import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Load the image and convert to grayscale
image = rgb2gray(data.astronaut())

# Function to calculate PSNR and SSIM
def calculate_metrics(original, compressed):
    psnr_value = psnr(original, compressed)
    ssim_value = ssim(original, compressed, data_range=1.0)
    return psnr_value, ssim_value

# Lists to store PSNR and SSIM values for CUR and SVD
psnr_cur = []
ssim_cur = []
psnr_svd = []
ssim_svd = []

# Function to apply CUR decomposition
def cur_decomposition(matrix, k):
    start_time = time()
    col_indices = np.random.choice(matrix.shape[1], k, replace=False)
    row_indices = np.random.choice(matrix.shape[0], k, replace=False)
    C = matrix[:, col_indices]
    R = matrix[row_indices, :]
    U = pinv(C) @ matrix @ pinv(R)
    cur_image = C @ U @ R
    elapsed_time = time() - start_time
    return cur_image, elapsed_time

# Function to apply SVD decomposition and reconstruct with top k singular values
def svd_reconstruction(matrix, k):
    start_time = time()
    U, S, Vt = svd(matrix, full_matrices=False)
    S_k = diagsvd(S[:k], k, k)
    svd_image = U[:, :k] @ S_k @ Vt[:k, :]
    elapsed_time = time() - start_time
    return svd_image, elapsed_time

# Set different ranks for demonstration
k_values = [20, 50, 100]
cur_times = []
svd_times = []

fig, axes = plt.subplots(2, len(k_values) + 1, figsize=(15, 10))

# Show original image
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')
axes[1, 0].imshow(image, cmap='gray')
axes[1, 0].set_title('Original Image')
axes[1, 0].axis('off')



# Show compressed images for each k and record times
for i, k in enumerate(k_values):
    # CUR Compression
    cur_compressed, cur_time = cur_decomposition(image, k)
    cur_times.append(cur_time)
    axes[0, i + 1].imshow(cur_compressed, cmap='gray')
    axes[0, i + 1].set_title(f'CUR k={k}')
    axes[0, i + 1].axis('off')
    
    # SVD Compression
    svd_compressed, svd_time = svd_reconstruction(image, k)
    svd_times.append(svd_time)
    axes[1, i + 1].imshow(svd_compressed, cmap='gray')
    axes[1, i + 1].set_title(f'SVD k={k}')
    axes[1, i + 1].axis('off')

    # Calculate PSNR and SSIM for CUR
    psnr_value_cur, ssim_value_cur = calculate_metrics(image, cur_compressed)
    psnr_cur.append(psnr_value_cur)
    ssim_cur.append(ssim_value_cur)
    
    # Calculate PSNR and SSIM for SVD
    psnr_value_svd, ssim_value_svd = calculate_metrics(image, svd_compressed)
    psnr_svd.append(psnr_value_svd)
    ssim_svd.append(ssim_value_svd)

plt.tight_layout()
plt.savefig("imcomp.png")
plt.close()

# Plot computational times
fig, ax = plt.subplots()
ax.plot(k_values, cur_times, label='CUR Decomposition', marker='o')
ax.plot(k_values, svd_times, label='SVD Decomposition', marker='o')
ax.set_xlabel('Rank k')
ax.set_ylabel('Time (seconds)')
ax.set_title('Computational Time Comparison')
ax.legend()

plt.savefig("imcomptime.png")

fig, axes = plt.subplots(2, 1, figsize=(8, 8))

axes[0].plot(k_values, psnr_cur, label='CUR', marker='o')
axes[0].plot(k_values, psnr_svd, label='SVD', marker='o')
axes[0].set_xlabel('Rank (k)')
axes[0].set_ylabel('PSNR')
axes[0].set_title('PSNR Comparison')
axes[0].legend()

axes[1].plot(k_values, ssim_cur, label='CUR', marker='o')
axes[1].plot(k_values, ssim_svd, label='SVD', marker='o')
axes[1].set_xlabel('Rank (k)')
axes[1].set_ylabel('SSIM')
axes[1].set_title('SSIM Comparison')
axes[1].legend()

plt.tight_layout()
plt.show()
