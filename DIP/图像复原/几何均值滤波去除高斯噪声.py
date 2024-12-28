import cv2
import numpy as np

# 读取灰度图像
image = cv2.imread('school_gray.jpg', cv2.IMREAD_GRAYSCALE)

# 添加高斯噪声函数
def add_gaussian_noise(img, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, img.shape)  # 生成高斯噪声
    noisy_image = img + noise  # 将噪声加到图像上
    noisy_image = np.clip(noisy_image, 0, 255)  # 将像素值限制在[0, 255]范围内
    return noisy_image.astype(np.uint8)

# 几何均值滤波器函数
def geometric_mean_filter(img, kernel_size=3):
    padded_img = np.pad(img, ((kernel_size // 2, ), (kernel_size // 2, )), mode='constant', constant_values=1)
    filtered_img = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # 获取当前窗口的像素值，计算其几何均值
            window = padded_img[i:i + kernel_size, j:j + kernel_size]
            geo_mean = np.exp(np.sum(np.log(window + 1e-8)) / (kernel_size ** 2))  # 计算几何均值
            filtered_img[i, j] = geo_mean
    return np.clip(filtered_img, 0, 255).astype(np.uint8)

# 添加高斯噪声到图像
noisy_image = add_gaussian_noise(image)

# 使用几何均值滤波器处理带噪声的图像
filtered_image = geometric_mean_filter(noisy_image)

# 显示原图、带噪声的图像和滤波后图像
cv2.imshow("Original Image", image)
cv2.imshow("Noisy Image", noisy_image)
cv2.imshow("Filtered Image", filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
