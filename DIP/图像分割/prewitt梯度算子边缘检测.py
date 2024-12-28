import cv2
import numpy as np

# 读取灰度图像
image = cv2.imread('school_gray.jpg', cv2.IMREAD_GRAYSCALE)

# 定义Prewitt算子的水平和垂直核
prewitt_kernel_x = np.array([[1, 0, -1],
                             [1, 0, -1],
                             [1, 0, -1]])

prewitt_kernel_y = np.array([[1, 1, 1],
                             [0, 0, 0],
                             [-1, -1, -1]])

# 应用卷积来计算水平和垂直方向的梯度
grad_x = cv2.filter2D(image, -1, prewitt_kernel_x)
grad_y = cv2.filter2D(image, -1, prewitt_kernel_y)

# 计算梯度幅值
gradient_magnitude = cv2.magnitude(grad_x.astype(np.float32), grad_y.astype(np.float32))
gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

# 显示原图和边缘检测结果
cv2.imshow("Original Image", image)
cv2.imshow("Prewitt Edge Detection", gradient_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()
