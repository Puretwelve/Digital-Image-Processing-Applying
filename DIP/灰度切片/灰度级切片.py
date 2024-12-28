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
    # 定义转化函数 T(r)
    def thresholding(gray_image, low, high):
        # 创建一个空白图像
        output_image = np.zeros_like(gray_image)
        # 应用转化函数
        output_image[(gray_image >= low) & (gray_image <= high)] = 255
        return output_image

    # 应用灰度级切片
    low_threshold = 100
    high_threshold = 150
    sliced_image = thresholding(gray_image, low_threshold, high_threshold)

    # 显示处理后的图片
    cv2.imshow('Sliced Image', sliced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 显示转化函数图
    r = np.arange(256)
    T = np.where((r >= low_threshold) & (r <= high_threshold), 255, 0)
    plt.plot(r, T, label='T(r)')
    plt.xlabel('Input Gray Level')
    plt.ylabel('Output Gray Level')
    plt.title('Thresholding Function T(r)')
    plt.legend()
    plt.grid(True)
    plt.show()