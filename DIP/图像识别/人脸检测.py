import cv2

# 1. 加载人脸检测的 Haar Cascade 分类器模型
# OpenCV 提供了一个预训练的模型 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. 读取输入图像
img = cv2.imread('face.jpg')
if img is None:
    print("无法加载图像，请检查文件路径")
    exit()

# 3. 转换为灰度图像，因为 Viola-Jones 需要灰度图像进行处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4. 进行人脸检测
# 参数依次是：灰度图像、检测的最小目标大小、邻近的矩形区域之间的最小间隔等
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 5. 在图像上绘制矩形框标记出人脸
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色矩形框，线宽为2

# 6. 显示检测结果
cv2.imshow('Detected Faces', img)

# 等待按键操作，关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
