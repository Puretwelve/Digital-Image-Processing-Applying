import cv2
import numpy as np

# 读取灰度图像
image = cv2.imread('school_gray.jpg', cv2.IMREAD_GRAYSCALE)

# 高斯平滑处理
# 设置高斯核大小（这里选择5x5）以及标准差（sigmaX，sigmaY可设置为0让函数自动计算）
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# 定义Prewitt算子的水平和垂直核
prewitt_kernel_x = np.array([[1, 0, -1],
                             [1, 0, -1],
                             [1, 0, -1]])

prewitt_kernel_y = np.array([[1, 1, 1],
                             [0, 0, 0],
                             [-1, -1, -1]])

# 应用卷积来计算水平和垂直方向的梯度
grad_x = cv2.filter2D(blurred_image, -1, prewitt_kernel_x)
grad_y = cv2.filter2D(blurred_image, -1, prewitt_kernel_y)

# 计算梯度幅值
gradient_magnitude = cv2.magnitude(grad_x.astype(np.float32), grad_y.astype(np.float32))
gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

# 显示原图、平滑后的图以及边缘检测结果
cv2.imshow("Original Image", image)
cv2.imshow("Blurred Image", blurred_image)
cv2.imshow("Prewitt Edge Detection after Blurring", gradient_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()