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
    # 应用高斯滤波进行平滑处理
    sigma = 1.0  # 高斯滤波的标准差
    smoothed_image = cv2.GaussianBlur(gray_image, (0, 0), sigma)

    # 定义拉普拉斯算子
    laplacian_kernel = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]], dtype=np.float32)

    # 应用拉普拉斯算子
    laplacian_image = cv2.filter2D(smoothed_image, -1, laplacian_kernel)

    # 生成锐化图像，通过将原图与拉普拉斯边缘图相加
    sharpened_image = np.clip(gray_image + laplacian_image, 0, 255).astype(np.uint8)

    # 创建显示结果的图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 显示原始图像
    axes[0].imshow(gray_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 显示拉普拉斯边缘图
    axes[1].imshow(laplacian_image, cmap='gray')
    axes[1].set_title('Laplacian Edge')
    axes[1].axis('off')

    # 显示锐化后的图像
    axes[2].imshow(sharpened_image, cmap='gray')
    axes[2].set_title('Sharpened Image with Laplacian')
    axes[2].axis('off')

    # 展示结果
    plt.tight_layout()
    plt.show()
