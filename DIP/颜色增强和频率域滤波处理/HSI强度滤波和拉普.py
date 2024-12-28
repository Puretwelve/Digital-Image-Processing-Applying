import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter, convolve

# 加载HSI图像（假设已经保存为强度分量图像）
hsi_image_path = r"D:\Digital Image Processing\HSI_image.png"
hsi_image = Image.open(hsi_image_path)
hsi_image_rgb = np.array(hsi_image)

# 提取强度通道（假设图像已经是HSI格式，且强度在第三个通道）
intensity = hsi_image_rgb[:, :, 2]  # 通道索引可能需要调整

# 应用均值滤波
def apply_mean_filter(image, size=3):
    return uniform_filter(image, size=size, mode='nearest')

# 应用拉普拉斯算子
def apply_laplacian(image):
    laplacian_kernel = np.array([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]])
    return convolve(image, laplacian_kernel)

# 对强度通道进行均值滤波
mean_filtered_intensity = apply_mean_filter(intensity)

# 对强度通道进行拉普拉斯变换
laplacian_intensity = apply_laplacian(intensity)

# 创建显示结果的图像
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 显示原始强度图像
axes[0].imshow(intensity, cmap='gray')
axes[0].set_title('Original Intensity')
axes[0].axis('off')

# 显示均值滤波后的强度图像
axes[1].imshow(mean_filtered_intensity, cmap='gray')
axes[1].set_title('Mean Filtered Intensity')
axes[1].axis('off')

# 显示拉普拉斯变换后的强度图像
axes[2].imshow(laplacian_intensity, cmap='gray')
axes[2].set_title('Laplacian Intensity')
axes[2].axis('off')

# 调整布局
plt.tight_layout()
plt.show()
