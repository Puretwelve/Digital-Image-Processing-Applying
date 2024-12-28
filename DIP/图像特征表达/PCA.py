import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像（灰度图像）
image = cv2.imread("school_gray2.jpg", cv2.IMREAD_GRAYSCALE)

# 确保图像大小为 512x512
image = cv2.resize(image, (512, 512))

# 2. 将图像数据转换为 2D 数组，每一行是图像的一行像素
h, w = image.shape
A = image.astype(np.float32)

# 3. 计算协方差矩阵
A_mean = np.mean(A, axis=0)  # 每列的均值
A_centered = A - A_mean  # 中心化

# 4. 进行特征值分解
cov_matrix = np.cov(A_centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # 特征值和特征向量

# 5. 排序特征值，并选取前50个最大的特征值
sorted_indices = np.argsort(eigenvalues)[::-1]  # 降序排序
top_k = 50  # 选择前50个特征值
eigenvalues_top50 = eigenvalues[sorted_indices[:top_k]]
eigenvectors_top50 = eigenvectors[:, sorted_indices[:top_k]]

# 6. 构造新的对角矩阵Λ，其中前50个特征值保留，其他填0
Lambda_top50 = np.zeros_like(cov_matrix)
np.fill_diagonal(Lambda_top50, eigenvalues_top50)

# 7. 使用前50个特征向量和特征值构造降维后的图像
A_reduced = np.dot(A_centered, eigenvectors_top50)  # 投影到前50个主成分上

# 8. 恢复图像：通过特征向量和特征值恢复
A_reconstructed = np.dot(A_reduced, eigenvectors_top50.T) + A_mean

# 9. 转换恢复后的图像为 uint8 类型（像素值范围0-255）
A_reconstructed = np.clip(A_reconstructed, 0, 255).astype(np.uint8)

# 10. 显示原始图像和恢复图像对比
plt.figure(figsize=(12, 6))

# 显示原始图像
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 显示恢复后的图像
plt.subplot(1, 2, 2)
plt.imshow(A_reconstructed, cmap='gray')
plt.title(f'Reconstructed Image with {top_k} Components')
plt.axis('off')

plt.show()
