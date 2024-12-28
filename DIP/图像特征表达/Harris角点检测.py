import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载灰度图像 "school_gray2.jpg"
gray_image = cv2.imread('school_gray2.jpg', cv2.IMREAD_GRAYSCALE)  # 直接以灰度模式加载

# 2. 将灰度图像转换为浮动格式
gray_image = np.float32(gray_image)

# 3. 使用 Harris 角点检测
dst = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)

# 4. 对结果进行膨胀，以便角点更加显著
dst = cv2.dilate(dst, None)

# 5. 设置阈值，标记角点
threshold = 0.01 * dst.max()  # 选择阈值（最大值的 1%）
# 加载原始图像用于标记角点
image = cv2.imread('school_gray2.jpg')

# 6. 在角点位置标记为红色（对于灰度图像，将其颜色设置为白色）
image[dst > threshold] = [255, 0, 0]  # 由于是灰度图，标记为蓝色

# 7. 显示结果
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 转换为 RGB 格式显示
plt.title('Harris Corner Detection')
plt.axis('off')
plt.show()
