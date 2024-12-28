import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

# 1. 读取灰度图像
image = cv2.imread('school_gray.jpg', cv2.IMREAD_GRAYSCALE)

# 2. 添加椒盐噪声
# 使用skimage的random_noise函数，设置噪声模式为'pepper'和'salt'，噪声密度为0.05
noisy_image = random_noise(image, mode='s&p', amount=0.05)
noisy_image = np.array(255 * noisy_image, dtype=np.uint8)  # 转换为0-255的范围

# 3. 使用中值滤波器去除椒盐噪声
filtered_image = cv2.medianBlur(noisy_image, 3)  # 使用3x3的中值滤波器

# 4. 显示原始图像、噪声图像和滤波后的图像
plt.figure(figsize=(12, 6))

# 原始图像
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 添加椒盐噪声后的图像
plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Image with Salt and Pepper Noise')
plt.axis('off')

# 中值滤波后图像
plt.subplot(1, 3, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Image After Median Filtering')
plt.axis('off')

# 展示结果
plt.tight_layout()
plt.show()