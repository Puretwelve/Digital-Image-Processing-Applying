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

    # 返回合并的HSI图像（H通道为色相，S通道为饱和度，I通道为强度）
    hsi_image = np.zeros_like(rgb_image)
    hsi_image[:, :, 0] = hue / 180  # 归一化到[0, 1]范围
    hsi_image[:, :, 1] = saturation  # 饱和度
    hsi_image[:, :, 2] = intensity    # 强度

    return hsi_image

# 进行转换
hsi_image = rgb_to_hsi(image_rgb)

# 创建显示结果的图像
plt.figure(figsize=(8, 8))
plt.imshow(hsi_image)
plt.title('HSI Image')
plt.axis('off')
plt.show()
