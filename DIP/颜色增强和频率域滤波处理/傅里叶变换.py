import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 打开灰度图像
image_path = r"D:\Digital Image Processing\school_gray.jpg"
gray_image = Image.open(image_path).convert('L')  # 确保是灰度图像
gray_image = np.array(gray_image)

# 计算离散傅里叶变换
dft = np.fft.fft2(gray_image)  # 计算二维傅里叶变换
dft_shifted = np.fft.fftshift(dft)  # 将零频率成分移到频谱中心

# 计算频谱的幅度谱并取对数
magnitude_spectrum = np.log(np.abs(dft_shifted) + 1)  # 加1避免对数零值

# 进行傅里叶逆变换
inverse_dft = np.fft.ifft2(np.fft.ifftshift(dft_shifted))  # 先逆移，再逆变换
inverse_image = np.abs(inverse_dft)  # 取绝对值作为结果

# 创建显示结果的图像
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(inverse_image, cmap='gray')
plt.title('Reconstructed Image from Inverse DFT')
plt.axis('off')

# 调整布局并显示
plt.tight_layout()
plt.show()
