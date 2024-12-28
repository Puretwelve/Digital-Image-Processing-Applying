import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve

# 加载RGB图像
rgb_image_path = r"D:\Digital Image Processing\full color image.jpg"
rgb_image = Image.open(rgb_image_path)
rgb_image = np.array(rgb_image)

# 加载HSI图像（假设强度图像已经处理过）
hsi_image_path = r"D:\Digital Image Processing\HSI_image.png"
hsi_image = Image.open(hsi_image_path)
hsi_image = np.array(hsi_image)

# 提取强度通道（假设强度在第三个通道）
intensity = hsi_image[:, :, 2]

# 确保RGB图像和强度图像的大小相同
if rgb_image.shape[:2] != intensity.shape:
    intensity = np.array(Image.fromarray(intensity).resize(rgb_image.shape[1::-1]))

# 应用拉普拉斯算子
def apply_laplacian(image):
    laplacian_kernel = np.array([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]])
    return convolve(image, laplacian_kernel)

# 对RGB图像和强度通道分别应用拉普拉斯算子
laplacian_rgb = np.zeros_like(rgb_image)
for i in range(3):  # 对每个通道分别应用
    laplacian_rgb[:, :, i] = apply_laplacian(rgb_image[:, :, i])

laplacian_intensity = apply_laplacian(intensity)

# 计算两幅图像之间的差别
difference_image = np.abs(laplacian_rgb.astype(np.int16) - laplacian_intensity[:, :, np.newaxis].astype(np.int16))

# 创建显示结果的图像
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# 显示RGB拉普拉斯变换图像
axes[0].imshow(laplacian_rgb.astype(np.uint8))
axes[0].set_title('Laplacian of RGB Image')
axes[0].axis('off')

# 显示HSI强度图像的拉普拉斯变换图像
axes[1].imshow(laplacian_intensity, cmap='gray')
axes[1].set_title('Laplacian of Intensity Image')
axes[1].axis('off')

# 显示差别图像
axes[2].imshow(difference_image.astype(np.uint8))
axes[2].set_title('Difference Image')
axes[2].axis('off')

# 显示原始RGB图像
axes[3].imshow(rgb_image)
axes[3].set_title('Original RGB Image')
axes[3].axis('off')

# 调整布局
plt.tight_layout()
plt.show()
