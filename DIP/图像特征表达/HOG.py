import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure, io
import numpy as np

# 1. 加载自定义图像 "school.jpg"
image = io.imread('school.jpg')  # 这里替换成你的图像路径

# 2. 计算图像的 HOG 特征
fd, hog_image = hog(
    image,                # 输入图像
    orientations=8,       # 梯度方向数（即每个单元格中分为多少个梯度方向）
    pixels_per_cell=(16, 16),  # 每个单元格的像素大小
    cells_per_block=(1, 1),   # 每个块的单元格数
    visualize=True,       # 是否可视化 HOG 特征图
    channel_axis=-1,      # 使用图像的所有通道（适用于 RGB 彩色图像）
)

# 3. 创建两个子图：一个显示原始图像，另一个显示 HOG 特征图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

# 4. 显示原始图像
ax1.axis('off')  # 不显示坐标轴
ax1.imshow(image)  # 显示图像（RGB模式）
ax1.set_title('Input image')  # 设置标题

# 5. 对 HOG 特征图进行重缩放，使其显示效果更好
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# 6. 显示 HOG 特征图
ax2.axis('off')  # 不显示坐标轴
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)  # 显示 HOG 特征图（灰度模式）
ax2.set_title('Histogram of Oriented Gradients')  # 设置标题

# 7. 绘制 HOG 特征的直方图
plt.figure(figsize=(6, 4))
plt.hist(fd, bins=50, color='gray')  # fd 是 HOG 特征的直方图数据
plt.title('Histogram of HOG Features')
plt.xlabel('Orientation bin')
plt.ylabel('Frequency')
plt.show()
