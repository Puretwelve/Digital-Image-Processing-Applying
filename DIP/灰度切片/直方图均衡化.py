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
    # 计算原图像的直方图
    hist_original = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    hist_original = hist_original.flatten()

    # 进行直方图均衡化
    equalized_image = cv2.equalizeHist(gray_image)

    # 计算均衡化后图像的直方图
    hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
    hist_equalized = hist_equalized.flatten()

    # 创建显示直方图的图像
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 显示原始图像的直方图
    axes[0, 0].plot(hist_original, color='lightblue')
    axes[0, 0].set_title('Histogram of Original Image')
    axes[0, 0].set_xlim([0, 256])

    # 显示均衡化后图像的直方图
    axes[0, 1].plot(hist_equalized, color='lightblue')
    axes[0, 1].set_title('Histogram of Equalized Image')
    axes[0, 1].set_xlim([0, 256])

    # 显示原始图像
    axes[1, 0].imshow(gray_image, cmap='gray')
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')

    # 显示均衡化后图像
    axes[1, 1].imshow(equalized_image, cmap='gray')
    axes[1, 1].set_title('Equalized Image')
    axes[1, 1].axis('off')

    # 确保布局紧凑
    plt.tight_layout()
    plt.show()