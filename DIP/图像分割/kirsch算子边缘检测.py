import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 载入灰度图像
img_clean = cv.imread('school_gray.jpg', cv.IMREAD_GRAYSCALE)

# Kirsch 算子的8个方向卷积核
m1 = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
m2 = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
m3 = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
m4 = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
m5 = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
m6 = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
m7 = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
m8 = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])

filterlist = [m1, m2, m3, m4, m5, m6, m7, m8]  # 将各个方向的卷积核放到一起，便于统一操作

# 建立三维数组，第0维表示各个方向卷积后的值
filtered_list = np.zeros((8, img_clean.shape[0], img_clean.shape[1]))

# 对8个方向进行卷积处理
for k in range(8):
    out = cv.filter2D(img_clean, cv.CV_16S, filterlist[k])  # 自定义卷积操作
    filtered_list[k] = out

# 取八个方向中的最大值，作为图像该点的新的像素值
final = np.max(filtered_list, axis=0)
final[final >= 255] = 255  # 令像素值大于255的点等于255
final[final < 255] = 0  # 令像素值小于255的点等于0

# 显示原图和边缘检测结果
fig = plt.figure(figsize=(10, 5))
fig.set(alpha=0.2)
plt.subplot2grid((1, 2), (0, 0))
plt.imshow(img_clean, cmap='gray')  # 显示原始图像

plt.subplot2grid((1, 2), (0, 1))
plt.imshow(final, cmap='gray')  # 显示经过 Kirsch 算子处理后的图像

plt.show()