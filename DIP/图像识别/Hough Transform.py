import cv2
import numpy as np

# 读取图像（假设图像已经是灰度图像）
img = cv2.imread('school_gray.jpg', cv2.IMREAD_GRAYSCALE)  # 直接读取灰度图像
if img is None:
    print("无法加载图像")
    exit()

# 使用高斯模糊去噪声
gray_blurred = cv2.GaussianBlur(img, (3, 3), 1)

# Canny 边缘检测
edges = cv2.Canny(gray_blurred, 50, 200, apertureSize=3)

# 霍夫变换检测直线
lines = cv2.HoughLines(edges, 1, np.pi / 180, 140)

# 在图像上绘制检测到的直线
# 将灰度图像转换为彩色图像，以便绘制彩色直线
color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # 绘制直线
        cv2.line(color_img, (x1, y1), (x2, y2), (55, 100, 195), 2, cv2.LINE_AA)

# 显示结果
print(f"检测到的直线数量: {len(lines) if lines is not None else 0}")
cv2.imshow("Hough Line Transform", color_img)

# 等待键盘输入，关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
