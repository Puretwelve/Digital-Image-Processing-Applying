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
    # Sobel 卷积核（转换为OpenCV中符合的数据类型）
    sobel_kernel_x = np.array([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]], dtype=np.float32)
    sobel_kernel_y = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]], dtype=np.float32)

    # 应用Sobel算子（分别计算水平和垂直方向的梯度）
    gradient_x = cv2.filter2D(gray_image, -1, sobel_kernel_x)
    gradient_y = cv2.filter2D(gray_image, -1, sobel_kernel_y)

    # 计算边缘强度（合并水平和垂直方向梯度）
    sobel_edge = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    sobel_edge = np.uint8(sobel_edge)  # 将数据类型转换为适合显示的uint8类型

    # 叠加锐化图像
    sharp_image = np.clip(gray_image + sobel_edge, 0, 255)
    sharp_image = np.uint8(sharp_image)

    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gray_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(sobel_edge, cmap='gray')
    axes[1].set_title('Sobel Edge')
    axes[1].axis('off')

    axes[2].imshow(sharp_image, cmap='gray')
    axes[2].set_title('Sharpened Image with Sobel')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()