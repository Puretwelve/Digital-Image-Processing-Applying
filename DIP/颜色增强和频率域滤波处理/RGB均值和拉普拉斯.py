import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter, convolve

# 打开彩色图像
image_path = r"D:\Digital Image Processing\full color image.jpg"
image = Image.open(image_path)
image_rgb = np.array(image)

# 应用均值滤波
def apply_mean_filter(image, size=3):
    return uniform_filter(image, size=size, mode='nearest')

# 应用拉普拉斯算子
def apply_laplacian(image):
    laplacian_kernel = np.array([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]])
    return convolve(image, laplacian_kernel)

# 对每个通道分别应用均值滤波
mean_filtered_image = np.zeros_like(image_rgb)
for i in range(3):  # RGB通道
    mean_filtered_image[:, :, i] = apply_mean_filter(image_rgb[:, :, i])

# 对每个通道分别应用拉普拉斯算子
laplacian_image = np.zeros_like(image_rgb)
for i in range(3):  # RGB通道
    laplacian_image[:, :, i] = apply_laplacian(image_rgb[:, :, i])

# 将拉普拉斯图像归一化到0-255
laplacian_image = np.clip(laplacian_image, 0, 255)

# 创建显示结果的图像
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 显示原始图像
axes[0].imshow(image_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')

# 显示均值滤波后的图像
axes[1].imshow(mean_filtered_image.astype(np.uint8))
axes[1].set_title('Mean Filtered Image')
axes[1].axis('off')

# 显示拉普拉斯变换后的图像
axes[2].imshow(laplacian_image.astype(np.uint8))
axes[2].set_title('Laplacian Image')
axes[2].axis('off')

# 调整布局
plt.tight_layout()
plt.show()
