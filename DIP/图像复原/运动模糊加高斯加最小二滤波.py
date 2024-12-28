import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2, fftshift

# 读取灰度图像
image = cv2.imread('school_gray.jpg', cv2.IMREAD_GRAYSCALE)

# 添加运动模糊函数
def motion_blur(img, kernel_size=15):
    # 创建运动模糊核
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    # 应用运动模糊
    blurred = convolve2d(img, kernel, boundary='wrap', mode='same')
    return blurred

# 添加高斯噪声函数
def add_gaussian_noise(img, mean=0, sigma=0.25):
    noise = np.random.normal(mean, sigma, img.shape)  # 生成高斯噪声
    noisy_image = img + noise  # 将噪声加到图像上
    noisy_image = np.clip(noisy_image, 0, 255)  # 将像素值限制在[0, 255]范围内
    return noisy_image.astype(np.uint8)

# 维纳滤波器
def wiener_filter(img, kernel, K=0.01):
    # 傅里叶变换
    img_fft = fft2(img)
    kernel_fft = fft2(kernel, s=img.shape)
    # 维纳滤波公式
    kernel_fft_conj = np.conj(kernel_fft)
    wiener_filter_fft = (kernel_fft_conj / (np.abs(kernel_fft)**2 + K)) * img_fft
    restored_img = np.abs(ifft2(wiener_filter_fft))
    return np.clip(restored_img, 0, 255).astype(np.uint8)

# 约束最小二乘法滤波
def constrained_least_squares_filter(img, kernel, gamma=0.01):
    # 傅里叶变换
    img_fft = fft2(img)
    kernel_fft = fft2(kernel, s=img.shape)
    # 拉普拉斯算子，用于正则化
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplacian_fft = fft2(laplacian, s=img.shape)
    # 约束最小二乘滤波公式
    cls_filter_fft = (np.conj(kernel_fft) / (np.abs(kernel_fft)**2 + gamma * np.abs(laplacian_fft)**2)) * img_fft
    restored_img = np.abs(ifft2(cls_filter_fft))
    return np.clip(restored_img, 0, 255).astype(np.uint8)

# 为图像添加运动模糊和高斯噪声
motion_blurred_image = motion_blur(image)
noisy_blurred_image = add_gaussian_noise(motion_blurred_image)

# 创建运动模糊核
kernel_size = 15
motion_kernel = np.zeros((kernel_size, kernel_size))
motion_kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
motion_kernel /= kernel_size

# 使用维纳滤波器恢复图像
wiener_restored = wiener_filter(noisy_blurred_image, motion_kernel)

# 使用约束最小二乘法滤波恢复图像
cls_restored = constrained_least_squares_filter(noisy_blurred_image, motion_kernel)

# 显示原图、模糊带噪图像及恢复后的图像
cv2.imshow("Original Image", image)
cv2.imshow("Motion Blurred with Noise", noisy_blurred_image)
cv2.imshow("Wiener Filtered Image", wiener_restored)
cv2.imshow("Constrained Least Squares Filtered Image", cls_restored)
cv2.waitKey(0)
cv2.destroyAllWindows()
