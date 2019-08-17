#1、读取图像，并把图像转换为灰度图像并显示
import cv2
import dlib                     #人脸识别的库 dlib
import numpy as np
im = cv2.imread(r"C:\Users\hutao\Desktop\model_save\22.jpg")  #读取图片
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   #转换了灰度化
#cv2.axis("off")
#cv2.title("Input Image")
cv2.imshow(im_gray, cmap="gray")  #显示图片
cv2.show()