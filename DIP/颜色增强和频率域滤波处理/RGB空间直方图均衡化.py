import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 打开彩色图像
image_path = r"D:\Digital Image Processing\full color image.jpg"
image = Image.open(image_path)
image_rgb = np.array(image)

# 创建显示结果的图像
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 显示原始RGB图像
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('Original RGB Image')
axes[0, 0].axis('off')

# 显示原始RGB直方图
axes[0, 1].hist(image_rgb[:, :, 0].ravel(), bins=256, color='red', alpha=0.5, label='Red Channel')
axes[0, 1].hist(image_rgb[:, :, 1].ravel(), bins=256, color='green', alpha=0.5, label='Green Channel')
axes[0, 1].hist(image_rgb[:, :, 2].ravel(), bins=256, color='blue', alpha=0.5, label='Blue Channel')
axes[0, 1].set_title('Original RGB Histogram')
axes[0, 1].set_xlim([0, 256])
axes[0, 1].legend()

# 直方图均衡化
def histogram_equalization(rgb_image):
    equalized_image = np.zeros_like(rgb_image)
    for i in range(3):  # 对每个通道进行均衡化
        channel = rgb_image[:, :, i]
        histogram, bins = np.histogram(channel.flatten(), bins=256, range=[0, 256])
        cdf = histogram.cumsum()  # 计算累积分布函数
        cdf_normalized = cdf * 255 / cdf[-1]  # 归一化
        equalized_image[:, :, i] = np.interp(channel.flatten(), bins[:-1], cdf_normalized).reshape(channel.shape)
    return equalized_image.astype(np.uint8)

# 进行均衡化处理
equalized_rgb_image = histogram_equalization(image_rgb)

# 显示均衡化后的RGB图像
axes[1, 0].imshow(equalized_rgb_image)
axes[1, 0].set_title('Equalized RGB Image')
axes[1, 0].axis('off')

# 显示均衡化后的RGB直方图
axes[1, 1].hist(equalized_rgb_image[:, :, 0].ravel(), bins=256, color='red', alpha=0.5, label='Red Channel')
axes[1, 1].hist(equalized_rgb_image[:, :, 1].ravel(), bins=256, color='green', alpha=0.5, label='Green Channel')
axes[1, 1].hist(equalized_rgb_image[:, :, 2].ravel(), bins=256, color='blue', alpha=0.5, label='Blue Channel')
axes[1, 1].set_title('Equalized RGB Histogram')
axes[1, 1].set_xlim([0, 256])
axes[1, 1].legend()

# 调整布局
plt.tight_layout()
plt.show()
