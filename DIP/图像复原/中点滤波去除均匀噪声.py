import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
image = cv2.imread('school_gray.jpg', cv2.IMREAD_GRAYSCALE)

# 检查图像是否加载成功
if image is None:
    raise ValueError("图片加载失败，请检查文件路径")

# 设置噪声的比例
noise_ratio = 0.1  # 10% 噪声


# 加入均匀噪声
def add_uniform_noise(img, noise_ratio):
    noisy_image = img.copy()
    total_pixels = img.size
    noise_pixels = int(total_pixels * noise_ratio)

    # 随机生成噪声像素位置
    noise_coords = np.random.randint(0, total_pixels, noise_pixels)
    for i in noise_coords:
        row = i // img.shape[1]
        col = i % img.shape[1]
        noisy_image[row, col] = np.random.randint(0, 256)  # 生成 0 到 255 的均匀噪声

    return noisy_image


# 添加噪声
noisy_image = add_uniform_noise(image, noise_ratio)


# 中点滤波器处理
def median_filter(img, kernel_size=3):
    return cv2.medianBlur(img, kernel_size)


# 使用中点滤波器去噪
denoised_image = median_filter(noisy_image)

# 显示原图、噪声图和去噪图
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Noisy Image')
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Denoised Image (Median Filter)')
plt.imshow(denoised_image, cmap='gray')
plt.axis('off')

plt.show()