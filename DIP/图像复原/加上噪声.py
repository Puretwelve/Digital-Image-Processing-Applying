import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

# 1. 加载原始灰度图像
image = cv2.imread('school_gray.jpg', cv2.IMREAD_GRAYSCALE)

# 2. 添加高斯噪声
def add_gaussian_noise(img, mean=0, var=0.01):
    row, col = img.shape
    sigma = var**0.5
    gaussian = np.random.normal(mean, sigma, (row, col))
    noisy_img = np.clip(img + gaussian * 255, 0, 255)
    return noisy_img.astype(np.uint8)

# 3. 添加均匀噪声
def add_uniform_noise(img, low=0, high=0.1):
    row, col = img.shape
    uniform = np.random.uniform(low, high, (row, col))
    noisy_img = np.clip(img + uniform * 255, 0, 255)
    return noisy_img.astype(np.uint8)

# 4. 添加椒盐噪声
def add_salt_and_pepper_noise(img, amount=0.05):
    noisy_img = random_noise(img, mode='s&p', amount=amount) * 255
    return noisy_img.astype(np.uint8)

# 5. 生成噪声图像
gaussian_noisy_img = add_gaussian_noise(image)
uniform_noisy_img = add_uniform_noise(image)
salt_pepper_noisy_img = add_salt_and_pepper_noise(image)

# 6. 显示原图和加噪图像
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# 原图
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# 高斯噪声图像
axes[0, 1].imshow(gaussian_noisy_img, cmap='gray')
axes[0, 1].set_title('Gaussian Noise')
axes[0, 1].axis('off')

# 均匀噪声图像
axes[1, 0].imshow(uniform_noisy_img, cmap='gray')
axes[1, 0].set_title('Uniform Noise')
axes[1, 0].axis('off')

# 椒盐噪声图像
axes[1, 1].imshow(salt_pepper_noisy_img, cmap='gray')
axes[1, 1].set_title('Salt & Pepper Noise')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# 7. 显示原图和加噪图像的直方图
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# 原图直方图
axes[0, 0].hist(image.ravel(), bins=256, color='black', alpha=0.7)
axes[0, 0].set_title('Original Image Histogram')

# 高斯噪声直方图
axes[0, 1].hist(gaussian_noisy_img.ravel(), bins=256, color='black', alpha=0.7)
axes[0, 1].set_title('Gaussian Noise Histogram')

# 均匀噪声直方图
axes[1, 0].hist(uniform_noisy_img.ravel(), bins=256, color='black', alpha=0.7)
axes[1, 0].set_title('Uniform Noise Histogram')

# 椒盐噪声直方图
axes[1, 1].hist(salt_pepper_noisy_img.ravel(), bins=256, color='black', alpha=0.7)
axes[1, 1].set_title('Salt & Pepper Noise Histogram')

plt.tight_layout()
plt.show()