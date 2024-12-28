import cv2
import numpy as np

# 添加高斯噪声
def add_gaussian_noise(image, mean=0, std=25):
    """向图像添加高斯噪声"""
    row, col = image.shape
    gauss = np.random.normal(mean, std, (row, col))  # 生成高斯噪声
    noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)  # 添加噪声并限制像素值在0-255之间
    return noisy_image

# 算术均值滤波器
def arithmetic_mean_filter(image, kernel_size=3):
    """算术均值滤波器"""
    return cv2.blur(image, (kernel_size, kernel_size))  # 使用 OpenCV 的均值滤波函数

# 读取图像
image = cv2.imread('school_gray.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Image not found!")
    exit()

# 添加高斯噪声
noisy_image = add_gaussian_noise(image, mean=0, std=25)  # 这里可以根据需要调整噪声的均值和标准差

# 使用算术均值滤波器去噪
filtered_image = arithmetic_mean_filter(noisy_image, kernel_size=3)  # 可以调整卷积核的大小

# 显示结果
cv2.imshow("Original Image", image)
cv2.imshow("Noisy Image", noisy_image)
cv2.imshow("Filtered Image", filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
