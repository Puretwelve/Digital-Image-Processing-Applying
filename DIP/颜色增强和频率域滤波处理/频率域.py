import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
image_path = 'school_gray.jpg'  # 替换为你的图像路径
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if gray_image is None:
    print("Error: Image not found or unable to load.")
    exit()

# --------------------- 空间域拉普拉斯算子 ---------------------
# 使用拉普拉斯算子进行边缘检测
laplacian_kernel = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]], dtype=np.float32)

# 应用拉普拉斯算子
laplacian_image = cv2.filter2D(gray_image, -1, laplacian_kernel)

# 从原始图像中减去拉普拉斯算子部分
enhanced_image_spatial = cv2.subtract(np.uint8(gray_image), np.uint8(laplacian_image))

# --------------------- 频率域拉普拉斯算子 ---------------------
# 计算图像的频谱
dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# 创建拉普拉斯滤波器
rows, cols = gray_image.shape
crow, ccol = rows // 2, cols // 2
laplacian_filter = np.zeros((rows, cols, 2), np.float32)

# 构建拉普拉斯滤波器
for x in range(cols):
    for y in range(rows):
        laplacian_filter[y, x] = -4 * (np.pi ** 2) * ((x - ccol) ** 2 + (y - crow) ** 2)

# 应用拉普拉斯滤波器
filtered_dft = dft_shift * laplacian_filter
filtered_idft = cv2.idft(np.fft.ifftshift(filtered_dft))
laplacian_freq_image = cv2.magnitude(filtered_idft[:, :, 0], filtered_idft[:, :, 1])

# 从原始图像中减去频域拉普拉斯部分
enhanced_image_freq = cv2.subtract(np.uint8(gray_image), np.uint8(laplacian_freq_image))

# 对增强图像进行归一化
enhanced_image_spatial = cv2.normalize(enhanced_image_spatial, None, 0, 255, cv2.NORM_MINMAX)
enhanced_image_freq = cv2.normalize(enhanced_image_freq, None, 0, 255, cv2.NORM_MINMAX)

# 转换为无符号8位整型
enhanced_image_spatial = np.uint8(enhanced_image_spatial)
enhanced_image_freq = np.uint8(enhanced_image_freq)

# --------------------- 显示结果 ---------------------
plt.figure(figsize=(12, 8))

# 原图像
plt.subplot(2, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 空间域拉普拉斯
plt.subplot(2, 3, 2)
plt.imshow(laplacian_image, cmap='gray')
plt.title('Spatial Domain Laplacian')
plt.axis('off')

# 空间域增强图像
plt.subplot(2, 3, 3)
plt.imshow(enhanced_image_spatial, cmap='gray')
plt.title('Enhanced Image (Spatial Domain)')
plt.axis('off')

# 频率域拉普拉斯
plt.subplot(2, 3, 4)
plt.imshow(laplacian_freq_image, cmap='gray')
plt.title('Frequency Domain Laplacian')
plt.axis('off')

# 频率域增强图像
plt.subplot(2, 3, 5)
plt.imshow(enhanced_image_freq, cmap='gray')
plt.title('Enhanced Image (Frequency Domain)')
plt.axis('off')

# 显示所有图像
plt.tight_layout()
plt.show()