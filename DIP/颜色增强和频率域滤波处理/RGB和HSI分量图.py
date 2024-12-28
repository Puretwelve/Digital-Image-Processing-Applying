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
    saturation = 1 - (min_rgb / (intensity + 1e-10))  # 加上小常数避免除零

    # 计算色相
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    theta = np.arccos(num / (den + 1e-10))  # 加上小常数避免除零

    # 根据 RGB 的值确定 H 的范围
    hue = np.zeros_like(intensity)
    hue[b <= g] = theta[b <= g]
    hue[b > g] = 2 * np.pi - theta[b > g]

    # 将H从弧度转换为度
    hue = hue * (180 / np.pi)

    # 返回HSI
    return hue, saturation, intensity


hue, saturation, intensity = rgb_to_hsi(image_rgb)

# 创建显示结果的图像
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 显示RGB分量图
axes[0, 0].imshow(image_rgb[:, :, 0], cmap='Reds')
axes[0, 0].set_title('Red Channel')
axes[0, 0].axis('off')

axes[0, 1].imshow(image_rgb[:, :, 1], cmap='Greens')
axes[0, 1].set_title('Green Channel')
axes[0, 1].axis('off')

axes[0, 2].imshow(image_rgb[:, :, 2], cmap='Blues')
axes[0, 2].set_title('Blue Channel')
axes[0, 2].axis('off')

# 显示HSI分量图
axes[1, 0].imshow(hue, cmap='hsv')
axes[1, 0].set_title('Hue Channel')
axes[1, 0].axis('off')

axes[1, 1].imshow(saturation, cmap='gray')
axes[1, 1].set_title('Saturation Channel')
axes[1, 1].axis('off')

axes[1, 2].imshow(intensity, cmap='gray')
axes[1, 2].set_title('Intensity Channel')
axes[1, 2].axis('off')

# 展示结果
plt.tight_layout()
plt.show()
