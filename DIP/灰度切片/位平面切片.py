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
    # 获取图像的位平面
    def bit_plane_slicing(image):
        bit_planes = []
        for i in range(8):
            # 提取第i个位平面
            bit_plane = (image >> i) & 1
            # 将其扩展为8位灰度图像
            bit_plane_image = bit_plane * 255
            bit_planes.append(bit_plane_image)
        return bit_planes

    # 应用位平面切片
    bit_planes = bit_plane_slicing(gray_image)

    # 创建一个3x3的网格来显示图像
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

    # 显示原始图像
    axes[0, 0].imshow(gray_image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 显示8个位平面，从最高位到最低位
    for i in range(8):
        row = (8 - i) // 3
        col = (8 - i) % 3
        axes[row, col].imshow(bit_planes[i], cmap='gray')
        axes[row, col].set_title(f'Bit Plane {7 - i}')
        axes[row, col].axis('off')

    # 确保布局紧凑
    plt.tight_layout()
    plt.show()