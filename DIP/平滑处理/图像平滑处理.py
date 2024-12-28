import cv2
import numpy as np
import matplotlib.pyplot as plt

# 图片路径
image_path = r"D:\Digital Image Processing\school_gray.jpg"

# 读取灰度图像
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功读取
if gray_image is None:
    print("无法加载图像，请检查文件路径。")
else:
    # 定义滤波器参数
    size = 5  # 滤波器窗口大小
    sigma = 1.0  # 高斯滤波器的标准差

    # 应用均值滤波器
    mean_filtered_image = cv2.blur(gray_image, (size, size))

    # 应用方框滤波器（在此示例中与均值滤波器相同）
    box_filtered_image = cv2.blur(gray_image, (size, size))

    # 应用高斯滤波器
    gaussian_filtered_image = cv2.GaussianBlur(gray_image, (0, 0), sigma)

    # 创建显示结果的图像
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 显示原始图像
    axes[0, 0].imshow(gray_image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 显示均值滤波后的图像
    axes[0, 1].imshow(mean_filtered_image, cmap='gray')
    axes[0, 1].set_title('Mean Filtered Image')
    axes[0, 1].axis('off')

    # 显示方框滤波后的图像
    axes[1, 0].imshow(box_filtered_image, cmap='gray')
    axes[1, 0].set_title('Box Filtered Image')
    axes[1, 0].axis('off')

    # 显示高斯滤波后的图像
    axes[1, 1].imshow(gaussian_filtered_image, cmap='gray')
    axes[1, 1].set_title('Gaussian Filtered Image')
    axes[1, 1].axis('off')

    # 展示对比图
    plt.tight_layout()
    plt.show()