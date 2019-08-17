#encoding:utf-8
import numpy as np
import cv2
image = cv2.imread(r"C:\Users\hutao\Desktop\model_save\001.jpg")

#彩色转灰色
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray",gray)
cv2.waitKey(0)