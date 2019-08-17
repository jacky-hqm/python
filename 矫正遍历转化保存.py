#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os, sys
import math
import dlib
import numpy as np
from PIL import Image
import cv2

def eye_features(a):
    # dlib预测器

     # cv2读取图像
    img=cv2.imread(a)
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 人脸数rects
    rects = detector(img_gray, 0)

    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
        index = [37,46]
        tmp_points = []  # 存放嘴部的特征点
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            #pos = (point[0, 0], point[0, 1])
            if idx in index:
                pos = (point[0, 0], point[0, 1])  # 每个特征点的x和y
                tmp_points.append(pos)
        print(tmp_points)
        return tmp_points
# 计算两个坐标的距离
def Distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)
# 根据参数，求仿射变换矩阵和变换后的图像。
def ScaleRotateTranslate(image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)
# 根据所给的人脸图像，眼睛坐标位置，偏移比例，输出的大小，来进行裁剪。
def CropFace(image, eye_left=(0, 0), eye_right=(0, 0)):
    # get the direction  计算眼睛的方向。
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians  计算旋转的方向弧度。
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # distance between them  # 计算两眼之间的距离。
    #dist = Distance(eye_left, eye_right)
     # 原图像绕着左眼的坐标旋转。
    image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
    return image

if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r'C:\Users\hutao\PycharmProjects\AI\check\vision_work\face_quality\models\landmarks.dat')
    # 打开文件
    path = r"C:\Users\hutao\Desktop\123456"
    dirs = os.listdir(path)
    for file in dirs:
        print(path+file)
        image = Image.open(path+'/'+file)
        pp = []
        pp = eye_features(path+'/'+file)
        if  pp is not None :
            leftx = pp[0][0]
            lefty = pp[0][1]
            rightx =pp[1][0]
            righty =pp[1][1]

            centrex =int((leftx+rightx)/2)
            centrey =int((lefty+righty)/2)

            print(leftx,lefty,rightx,righty)
            print(centrex,centrey)
            CropFace(image, eye_left=(leftx,lefty), eye_right=(rightx, righty)).save(file)






