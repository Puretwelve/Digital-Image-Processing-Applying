import cv2

# 图片路径
image_path = r"D:\Digital Image Processing\school.jpg"

# 读取图片
image = cv2.imread(image_path)

# 检查图片是否成功读取
if image is None:
    print("无法加载图像，请检查文件路径。")
else:
    # 将图片转换为灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 显示灰度图片
    cv2.imshow('school_gray.jpg', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()