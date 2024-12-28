import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 打开彩色图像
image_path = r"D:\Digital Image Processing\full color image.jpg"
image = Image.open(image_path)
image_rgb = np.array(image)

# 转换为HSI
def rgb_to_hsi(rgb_image):
    rgb_image = rgb_image.astype(float) / 255.0  # 归一化到[0, 1]
    r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]

    # 计算强度
    intensity = (r + g + b) / 3.0

    # 计算饱和度
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = 1 - (min_rgb / (intensity + 1e-10))  # 避免除零

    # 计算色相
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    theta = np.arccos(num / (den + 1e-10))  # 避免除零

    hue = np.zeros_like(intensity)
    hue[b <= g] = theta[b <= g]
    hue[b > g] = 2 * np.pi - theta[b > g]
    hue = hue * (180 / np.pi)  # 从弧度转换为度

    return hue, saturation, intensity

# 获取HSI通道
hue, saturation, intensity = rgb_to_hsi(image_rgb)

# 创建显示结果的图像
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 显示强度通道
axes[0, 0].imshow(intensity, cmap='gray')
axes[0, 0].set_title('Intensity Channel')
axes[0, 0].axis('off')

# 显示原始强度直方图
axes[0, 1].hist(intensity.ravel(), bins=256, color='purple', alpha=0.5, label='Intensity Channel')
axes[0, 1].set_title('Intensity Histogram')
axes[0, 1].set_xlim([0, 1])
axes[0, 1].legend()

# 直方图均衡化
def histogram_equalization_intensity(intensity_image):
    # 对强度通道进行均衡化
    histogram, bins = np.histogram(intensity_image.flatten(), bins=256, range=[0, 1])
    cdf = histogram.cumsum()  # 计算累积分布函数
    cdf_normalized = cdf * 255 / cdf[-1]  # 归一化
    equalized_intensity = np.interp(intensity_image.flatten(), bins[:-1], cdf_normalized).reshape(intensity_image.shape)
    return equalized_intensity.astype(np.uint8)

# 进行均衡化处理
equalized_intensity = histogram_equalization_intensity(intensity)

# 显示均衡化后的强度图像
axes[1, 0].imshow(equalized_intensity, cmap='gray')
axes[1, 0].set_title('Equalized Intensity Image')
axes[1, 0].axis('off')

# 显示均衡化后的强度直方图
axes[1, 1].hist(equalized_intensity.ravel(), bins=256, color='purple', alpha=0.5, label='Equalized Intensity Channel')
axes[1, 1].set_title('Equalized Intensity Histogram')
axes[1, 1].set_xlim([0, 255])
axes[1, 1].legend()

# 调整布局
plt.tight_layout()
plt.show()
