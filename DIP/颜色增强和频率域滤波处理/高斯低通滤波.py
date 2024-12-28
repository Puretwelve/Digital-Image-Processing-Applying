import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 打开灰度图像
image_path = r"D:\Digital Image Processing\school_gray.jpg"
gray_image = Image.open(image_path).convert('L')
gray_image = np.array(gray_image)

# 计算离散傅里叶变换
dft = np.fft.fft2(gray_image)
dft_shifted = np.fft.fftshift(dft)

# 创建高斯低通滤波器
def gaussian_low_pass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    gaussian_filter = np.exp(-(x**2 + y**2) / (2 * (cutoff**2)))
    return gaussian_filter

# 设置不同的截止频率
cutoff_frequencies = [10, 30, 50, 70]
filtered_images = []

for cutoff in cutoff_frequencies:
    # 应用高斯低通滤波器
    gaussian_filter = gaussian_low_pass_filter(gray_image.shape, cutoff)
    filtered_dft = dft_shifted * gaussian_filter

    # 计算逆傅里叶变换
    inverse_dft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))
    reconstructed_image = np.abs(inverse_dft)
    filtered_images.append(reconstructed_image)

# 创建显示结果的图像
fig, axes = plt.subplots(1, len(cutoff_frequencies) + 1, figsize=(20, 5))

# 显示原始图像
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original Grayscale Image')
axes[0].axis('off')

# 显示不同截止频率下的滤波结果
for i, cutoff in enumerate(cutoff_frequencies):
    axes[i + 1].imshow(filtered_images[i], cmap='gray')
    axes[i + 1].set_title(f'GLPF(cutoff={cutoff})')
    axes[i + 1].axis('off')

# 调整布局并显示
plt.tight_layout()
plt.show()
