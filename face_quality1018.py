import os
import cv2
import math
import dlib
import time
import numpy as np
from PIL import Image
# import common_data
from sklearn.externals import joblib


# 使用hog进行降维，特征提取
def hog_extract(img):
    img = np.array(img)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgHeight, imgWidth = img.shape
    # print(img)
    # print(imgWidth,imgHeight)
    # print('hog',img.shape,'type',type(img))
    hog = cv2.HOGDescriptor((imgWidth, imgHeight), (16, 16), (8, 8), (8, 8), 9)
    desciptors = hog.compute(img, (1, 1), (0, 0))
    return desciptors  # , len(desciptors)


# 边缘检测     嘴巴的4点   人脸框 放大因子
def edge_check(ret_points, boxs, count):
    ret_points_finall = []
    X, Y = [], []
    for point in ret_points:
        X.append(point[0])
        Y.append(point[1])
    min_x, max_x = min(X), max(X)
    min_y, max_y = min(Y), max(Y)
    # 左上角
    rect_l_u_x_ori = min_x
    if rect_l_u_x_ori > boxs.left and rect_l_u_x_ori < boxs.right:  # 原坐标在人脸框内
        rect_l_u_x = int(ret_points[1][0] * count)  # 放大缩小倍数后的坐标
        if rect_l_u_x > boxs.left and rect_l_u_x < boxs.right:  # 还在人脸框内
            pass
        else:  # 经过放大或缩小后不再框内了，就保存原来的值
            rect_l_u_x = rect_l_u_x_ori
    else:
        return None

    # 左上角y
    rect_l_u_y_ori = min_y
    if rect_l_u_y_ori > boxs.up and rect_l_u_y_ori < boxs.down:  # 原坐标在人脸框内
        rect_l_u_y = int(ret_points[0][1] * count)
        if rect_l_u_y > boxs.up and rect_l_u_y < boxs.down:
            pass
        else:
            rect_l_u_y = rect_l_u_y_ori
    else:
        return None

    # 右上角x
    rect_r_u_x_ori = max_x
    if rect_r_u_x_ori > boxs.left and rect_r_u_x_ori < boxs.right:
        rect_r_u_x = int(ret_points[2][0] / count)
        if rect_r_u_x > boxs.left and rect_r_u_x < boxs.right:
            pass
        else:
            rect_r_u_x = rect_r_u_x_ori
    else:
        return None

    # 右上角y
    rect_r_u_y_ori = min_y
    if rect_r_u_y_ori > boxs.up and rect_r_u_y_ori < boxs.down:
        rect_r_u_y = int(ret_points[0][1] * count)
        if rect_r_u_y > boxs.up and rect_r_u_y < boxs.down:
            pass
        else:
            rect_r_u_y = rect_r_u_y_ori
    else:
        return None

    # 左下角x
    rect_l_d_x_ori = min_x
    if rect_l_d_x_ori > boxs.left and rect_l_d_x_ori < boxs.right:
        rect_l_d_x = int(ret_points[1][0] * count)
        if rect_l_d_x > boxs.left and rect_l_d_x < boxs.right:
            pass
        else:
            rect_l_d_x = rect_l_d_x_ori
    else:
        return None

    # 左下角y
    rect_l_d_y_ori = max_y
    if rect_l_d_y_ori > boxs.up and rect_l_d_y_ori < boxs.down:
        rect_l_d_y = int(ret_points[3][1] / count)
        if rect_l_d_y > boxs.up and rect_l_d_y < boxs.down:
            pass
        else:
            rect_l_d_y = rect_l_d_y_ori
    else:
        return None

    # 右下角x
    rect_r_d_x_ori = max_x
    if rect_r_d_x_ori > boxs.left and rect_r_d_x_ori < boxs.right:
        rect_r_d_x = int(ret_points[2][0] / count)
        if rect_r_d_x > boxs.left and rect_r_d_x < boxs.right:
            pass
        else:
            rect_r_d_x = rect_r_d_x_ori
    else:
        return None

    # 右下角y
    rect_r_d_y_ori = max_y
    if rect_r_d_y_ori > boxs.up and rect_r_d_y_ori < boxs.down:
        rect_r_d_y = int(ret_points[3][1] / count)
        if rect_r_d_y > boxs.up and rect_r_d_y < boxs.down:
            pass
        else:
            rect_r_d_y = rect_r_d_y_ori
    else:
        return None
    # 整理成矩形的格式
    ret_points_finall.append((rect_l_u_x, rect_l_u_y))  # 左上
    ret_points_finall.append((rect_r_u_x, rect_r_u_y))  # 右上
    ret_points_finall.append((rect_l_d_x, rect_l_d_y))  # 左下
    ret_points_finall.append((rect_r_d_x, rect_r_d_y))  # 右下
    return ret_points_finall


def points_68_to_5_mouth(boxs, landmarks):
    # 图片特征坐标
    index = [50, 52, 48, 54, 57]
    tmp_points = []  # 存放嘴部的特征点
    count = 0.97
    # 鼻子下方特征点  33
    nose_point = []
    for idx, point in enumerate(landmarks):
        # 68点的坐标
        if idx in index:
            pos = (point[0, 0], point[0, 1])  # 每个特征点的x和y
            tmp_points.append(pos)
        if idx == 33:
            pos = (point[0, 0], point[0, 1])
            nose_point.append(pos)
    # print(tmp_points)#48 50 52 54 57
    ret_points = []  # 嘴中 48 54 57
    ret_points.append((int((int(tmp_points[2][0]) + int(tmp_points[1][0])) / 2),
                       int((int(tmp_points[2][1]) + int(tmp_points[1][1])) / 2)))
    ret_points.append(tmp_points[0])  # 48左
    ret_points.append(tmp_points[3])  # 54右
    ret_points.append(tmp_points[4])  # 57下
    # print(ret_points)
    # 矩形框的点
    ret_points_finall = []
    X, Y = [], []
    for point in ret_points:
        X.append(point[0])
        Y.append(point[1])
    min_x, max_x = min(X), max(X)
    min_y, max_y = min(Y), max(Y)
    # 整理成矩形的格式
    ret_points_finall.append((min_x, min_y))  # 左上
    ret_points_finall.append((max_x, max_y))  # 右下
    # ret_points_finall.append((max_x, min_y))  # 右上
    # ret_points_finall.append((min_x, max_y))  # 左下

    # box = [0] * 4  # 人脸框的四边坐标
    # box[0], box[1], box[2], box[3] = boxs.left(), boxs.top(), boxs.right(), boxs.bottom()
    # ret_points_finall = edge_check(ret_points, boxs, count)
    if ret_points_finall is not None:
        return ret_points_finall, ret_points, nose_point
    else:
        return None, None, None


# 根据参数，求仿射变换矩阵和变换后的图像。
def ScaleRotateTranslate(image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx, ny = x, y = center  # 左眼的坐标
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


# 图像防射变换后坐标变换关系
def coordinate_conversion(ori_coo, center_coo, rotation):
    ori_x, ori_y = ori_coo
    center_x, center_y = center_coo
    changed_x = ori_x - center_x
    changed_y = ori_y - center_y
    rotation = (-rotation)
    changed_coo = (changed_x * math.cos(rotation) + changed_y * math.sin(rotation) + center_x,
                   -changed_x * math.sin(rotation) + changed_y * math.cos(rotation) + center_y)
    return changed_coo


# 仿射变换2
def rotate(image, angle, center=None, scale=1.0):  # 1
    image = np.array(image)
    (h, w) = image.shape[:2]  # 2
    if center is None:  # 3
        center = (w // 2, h // 2)  # 4
    M = cv2.getRotationMatrix2D(center, angle, scale)  # 5
    rotated = cv2.warpAffine(image, M, (w, h))  # 6
    return rotated  # 7


# 根据所给的人脸图像，眼睛坐标位置，偏移比例，输出的大小，来进行裁剪。
def CropFace(image, mouth_left=(0, 0), mouth_right=(0, 0), rect_letf_up=(0, 0), rect_right_down=(0, 0),
             nose_point=(None, None)):
    img_wide = image.shape[1]
    img_height = image.shape[0]
    image = Image.fromarray(image)
    # 计算嘴巴的方向。
    mouth_direction = (mouth_right[0] - mouth_left[0], mouth_right[1] - mouth_left[1])
    # 计算旋转的方向弧度。
    rotation = -math.atan2(float(mouth_direction[1]), float(mouth_direction[0]))

    # 原图像绕着左嘴的坐标旋转。
    # image = ScaleRotateTranslate(image, center=mouth_left, angle=rotation)
    image = rotate(image, math.degrees((-rotation)), mouth_left)
    changed_mouth_left = coordinate_conversion(ori_coo=rect_letf_up, center_coo=mouth_left, rotation=rotation)
    changed_mouth_right = coordinate_conversion(ori_coo=rect_right_down, center_coo=mouth_left, rotation=rotation)
    # 鼻子点不为空
    if nose_point[0] is not None:
        changed_nose_point = coordinate_conversion(ori_coo=nose_point, center_coo=mouth_left, rotation=rotation)
        # 扩大边缘
        # 嘴巴的高度
        mouth_height = changed_mouth_right[1] - changed_mouth_left[1]
        # 鼻子距离嘴巴的距离
        nose_mouth = changed_mouth_left[1] - changed_nose_point[1]
        # 边缘检测,左上角x
        l_u_x = changed_mouth_left[0]
        # print(l_u_x)
        if l_u_x > 0 and l_u_x < img_wide:
            l_u_x = l_u_x - mouth_height * 0.5
            if l_u_x > 0 and l_u_x < img_wide:
                pass
            else:
                l_u_x = changed_mouth_left[0]
        else:
            return None, None, None
        # 左上角y
        l_u_y = changed_mouth_left[1]
        if l_u_y > 0 and l_u_y < img_height:
            l_u_y = l_u_y - nose_mouth
            if l_u_y > 0 and l_u_y < img_height:
                pass
            else:
                l_u_y = changed_mouth_left[1]
        else:
            return None, None, None

        # 右下角x
        r_d_x = changed_mouth_right[0]
        if r_d_x > 0 and r_d_x < img_wide:
            r_d_x = r_d_x + mouth_height * 0.5
            if r_d_x > 0 and r_d_x < img_wide:
                pass
            else:
                r_d_x = changed_mouth_right[0]
        else:
            return None, None, None
        # 右下角y
        r_d_y = changed_mouth_right[1]
        if r_d_y > 0 and r_d_y < img_height:
            r_d_y = r_d_y + nose_mouth
            if r_d_y > 0 and r_d_y < img_height:
                pass
            else:
                r_d_y = changed_mouth_right[1]
        else:
            return None, None, None
        changed_mouth_left = (int(l_u_x), int(l_u_y))
        changed_mouth_right = (int(r_d_x), int(r_d_y))
        return image, changed_mouth_left, changed_mouth_right
    # 鼻子点为空
    if nose_point[0] is None:
        return image, changed_mouth_left, changed_mouth_right


# 调用人脸检测，读取单张图片
def face_detect_single_pic(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_wide = image.shape[1]
    img_height = image.shape[0]
    # 人脸数rects
    rects = detector(img_gray, 1)
    if rects is not None:
        return rects, img_wide, img_height
    else:
        return None, None, None


# 检测原图的人脸，并返回特征点
def face_vector(img):
    '''
    输入原图，输出人脸图
    :param img: 原图
    :return: 人脸图
    '''
    rects, img_wide, img_height = face_detect_single_pic(img)
    if len(rects) == 0:
        # print("没有检测到人脸")
        return None
    if rects:
        for box in rects:
            # 把人脸截取出来
            face_l_u_x, face_l_u_y, face_r_d_x, face_r_d_y = box.left(), box.top(), box.right(), box.bottom()
            # 获取特征点(相对于原图片的)
            # print("正在提取特征点……")
            Img_clone = img.copy()  # 把原图拷贝一份
            cropImg_clone = np.array(Img_clone)  # 把原图转为矩阵
            # 3、人脸截取出来
            img_face_face = cropImg_clone[face_l_u_y: face_r_d_y, face_l_u_x:face_r_d_x]
            return img_face_face
    else:
        return None


##############################################################################################
# 使用mtcnn提取特征点
def point_5(landmarks, image):
    temp_point = []
    img_wide, img_height = image.shape[1], image.shape[0]
    for idx, point in enumerate(landmarks):
        # print(landmarks)
        pos = (point[0], point[1])
        temp_point.append(pos)  # 左眼，右眼，鼻子（30），左嘴（48），右嘴（54）
    mouth_points = []
    mouth_points.append(temp_point[2])  # 鼻子（30） 0
    mouth_points.append(temp_point[3])  # 左嘴（48） 1
    mouth_points.append(temp_point[4])  # 右嘴（54） 2
    # 计算48与54两点之间的直线
    k = (mouth_points[1][1] - mouth_points[2][1]) / (mouth_points[1][0] - mouth_points[2][0])
    b = mouth_points[1][1] - k * mouth_points[1][0]
    # 求出鼻子到嘴巴连线的距离
    h = abs((k * mouth_points[0][0] + b - mouth_points[0][1]) / math.sqrt(1 + k ** 2)) / 2
    # 整理成矩形的形式
    ret_points_finall = []
    rect_l_u_x, rect_l_u_y, rect_r_d_x, rect_r_d_y = mouth_points[1][0], mouth_points[1][1], mouth_points[2][0], \
                                                     mouth_points[2][1]
    # 左上角x
    if rect_l_u_x > 0 and rect_l_u_x < img_wide:
        rect_l_u_x = mouth_points[1][0] - h
        if rect_l_u_x > 0 and rect_l_u_x < img_wide:
            pass
        else:
            rect_l_u_x = mouth_points[1][0]
    else:
        return None, None
    # 左上角y
    if rect_l_u_y > 0 and rect_l_u_y < img_height:
        rect_l_u_y = mouth_points[1][1] - h
        if rect_l_u_y > 0 and rect_l_u_y < img_height:
            pass
        else:
            rect_l_u_y = mouth_points[1][1]
    else:
        return None, None
    # 右下角x
    if rect_r_d_x > 0 and rect_r_d_x < img_wide:
        rect_r_d_x = mouth_points[2][0] + h
        if rect_r_d_x > 0 and rect_r_d_x < img_wide:
            pass
        else:
            rect_r_d_x = mouth_points[2][0]
    else:
        return None, None
    # 右下角y
    if rect_r_d_y > 0 and rect_r_d_y < img_height:
        rect_r_d_y = mouth_points[2][1] + h
        if rect_r_d_y > 0 and rect_r_d_y < img_height:
            pass
        else:
            rect_r_d_y = mouth_points[2][1]
    else:
        return None, None
    ret_points_finall.append((int(rect_l_u_x), int(rect_l_u_y)))  # 左上角  0
    ret_points_finall.append((int(rect_r_d_x), int(rect_r_d_y)))  # 右下角  1
    # 返回
    if ret_points_finall is not None:
        return ret_points_finall, mouth_points
    else:
        return None, None


class FaceQualityCheck():
    def __init__(self):
        model_dir, _ = os.path.split(os.path.realpath(__file__))
        # self.predictor = dlib.shape_predictor(os.path.join(model_dir, 'models/landmarks.dat'))
        #self.clf_mouse = joblib.load(os.path.join(model_dir, 'models/check_mouth.m'))
        #self.clf_face = joblib.load(os.path.join(model_dir, 'models/check_angle.m'))
        self.clf_face = joblib.load(os.path.join(model_dir, 'models/model.m'))

    def load_model(self, dlib_path, clf_mouse_path, clf_face_angle_path):
        # self.predictor = dlib.shape_predictor(dlib_path)
        #self.clf_mouse = joblib.load(clf_mouse_path)
        self.clf_face = joblib.load(clf_face_angle_path)

    # 嘴巴二分类
    def mouth_check(self, img_face_face, landmarks=None):
        '''
        输入人脸图，返回预测结果：0是有嘴，1是嘴部有遮挡
        :param img: 人脸图
        :return: y_pred；0是正脸，1是嘴部有遮挡
        '''
        try:
            # 有人脸也有特征点
            if img_face_face is not None and landmarks is not None:
                # start_time_vectors = time.time()
                mouth_points, ret_points = point_5(landmarks, img_face_face)
                # print("获取嘴巴特征点耗时%.4fms" % ((time.time() - start_time_vectors) * 1000))
                nose_point = None
            # 有人脸但是特征点需要自己获取
            elif img_face_face is not None and landmarks is None:
                # 把box转成rectangle
                box = dlib.rectangle(
                    **{"left": 0, "top": 0, "right": img_face_face.shape[1], "bottom": img_face_face.shape[0]})
                start_time_vectors = time.time()
                landmarks = np.matrix([[p.x, p.y] for p in self.predictor(img_face_face, box).parts()])  # 获取特征点
                print("landmarks predictor %.4fms" % ((time.time() - start_time_vectors) * 1000))
                mouth_points, ret_points, nose_point = points_68_to_5_mouth(box, landmarks)  # 获取嘴型的左上和右下(矫正之前的坐标)
            else:
                return None, 0

            if mouth_points is not None:
                mouth_leftux, mouth_leftuy, mouth_rightdx, mouth_rightdy = mouth_points[0][0], mouth_points[0][1], \
                                                                           mouth_points[1][0], mouth_points[1][1]
                # 获取在人脸图上，嘴巴48和54两点的坐标
                left_mou_x, left_mou_y, right_mou_x, right_mou_y = ret_points[1][0], ret_points[1][1], \
                                                                   ret_points[2][0], \
                                                                   ret_points[2][1]
                # 获取鼻子下方的坐标
                if nose_point is not None:
                    nose_x, nose_y = nose_point[0][0], nose_point[0][1]
                else:
                    nose_x, nose_y = None, None
                # 检测矫正耗时
                start_time_crop = time.time()
                mouth_jiao, mouth_left, mouth_right = CropFace(img_face_face,
                                                               mouth_left=(left_mou_x, left_mou_y),
                                                               mouth_right=(right_mou_x, right_mou_y),
                                                               rect_letf_up=(mouth_leftux, mouth_leftuy),
                                                               rect_right_down=(mouth_rightdx, mouth_rightdy),
                                                               nose_point=(nose_x, nose_y))
                print("CropFace耗时%.4fms" % ((time.time() - start_time_crop) * 1000))
                if mouth_jiao is not None:
                    # start_time_predmouth = time.time()
                    mouth_jiao = np.array(mouth_jiao)
                    cropImg_clone_jiao = mouth_jiao.copy()  # 把矫正后的人脸拷贝一份
                    cropImg_clone_jiao = np.array(cropImg_clone_jiao)
                    cropImg_mouth = cropImg_clone_jiao[int(mouth_left[1]):int(mouth_right[1]),
                                    int(mouth_left[0]):int(mouth_right[0])]  # 得到嘴巴
                    cropImg_mouth = cv2.resize(cropImg_mouth, (64, 32))

                    # global index_mouth
                    # index_mouth += 1
                    # cv2.imwrite(r'D:\svm\dlib\result\temp' + '\\' + 'none_mouth_test' + str(index_mouth) + '.jpg', cropImg_mouth)
                    # 画出矫正后的嘴型
                    # cv2.imshow('mouth_after', cropImg_mouth)
                    # cv2.waitKey(5)

                    # 预测
                    res1 = hog_extract(cropImg_mouth)
                    res1_1 = res1.reshape(1, -1)
                    res1_1 = np.array(res1_1)
                    # y_pred = clf_mouse.predict(res1_1)
                    y_pred = self.clf_mouse.predict_proba(res1_1)
                    act_label = np.argmax(np.array(y_pred))
                    # print("预测嘴巴耗时%.4fms" % ((time.time() - start_time_predmouth) * 1000))
                    # # 画出人脸框图
                    # face_l_u_x, face_l_u_y, face_r_d_x, face_r_d_y = box.left(), box.top(), box.right(), box.bottom()
                    # cv2.rectangle(img_face_face, (face_l_u_x, face_l_u_y), (face_r_d_x, face_r_d_y), (0, 255, 0), 2)
                    # img_face_face = cv2.putText(img_face_face, 'pred_result:' + str(y_pred), (face_l_u_x, face_l_u_y - 2),
                    #                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    # cv2.imshow('ori_image', img_face_face)
                    # cv2.waitKey(1)

                    # if y_pred == 0:  # 把原图保存为正脸
                    #     global index_frontal
                    #     index_frontal += 1
                    #     cv2.imwrite(r'D:\svm\dlib\result\own_camear_frontal' + '\\' + 'camear_frontal_' + str(index_frontal) + '.jpg', img_face_face)
                    # elif y_pred == 1:  # 把原图保存为大角度
                    #     global index_bigangle
                    #     index_bigangle += 1
                    #     cv2.imwrite(r'D:\svm\dlib\result\own_camear_bigangel' + '\\' + 'camear_bigangle_' + str(index_bigangle) + '.jpg', img_face_face)
                    # print('预测完成，人脸图已保存。')
                    # print("\n")
                    return act_label, y_pred[0][act_label]
                else:
                    return None, 0
            else:
                return None, 0
        except Exception:
            return None, 0

    # 人脸四分类
    def angle_check(self, img_face_face, clf_face):
        try:
            if img_face_face is None:
                print('read failed', img_face_face)
                return None
            res1 = cv2.resize(img_face_face, (64, 64))  # 对图片进行缩放，第一个参数是读入的图片，第二个是制定的缩放大小
            res1 = hog_extract(res1)
            res1_1 = res1.reshape(1, -1)
            res1_1 = np.array(res1_1)
            # y_pred = clf_face.predict(res1_1)
            y_pred = clf_face.predict_proba(res1_1)
            act_label = np.argmax(np.array(y_pred))
            return act_label, y_pred[0][act_label]
        except Exception:
            return None, 0

    # 质量判断方法
    # def evaluate_quality(self, tf, track_id=None, landmark=None):
    #     # 输入是人脸图片
    #     start = time.time()
    #     img_face = tf.face_mat
    #     img_big_face = tf.face_big_mat
    #     # mouth_label, mouth_score = self.mouth_check(img_face, landmark)
    #     mouth_label = 0
    #     mouth_score = 1.0
    #     print('mouth_check: %.4fms' % ((time.time() - start) * 1000))
    #     if mouth_label is not None:
    #         if mouth_label == 0:  # 有嘴
    #             start = time.time()
    #             pred_face, angle_score = self.angle_check(img_face, self.clf_face)
    #             print('angle_check ：%.4fms' % ((time.time() - start) * 1000))
    #             if pred_face is not None:
    #                 if pred_face == 0:
    #                     common_data.save_face_pic('angle_left', img_big_face, track_id + common_data.random_shortname())
    #                     return False, 0
    #                 elif pred_face == 1:
    #                     common_data.save_face_pic('angle_right', img_big_face,
    #                                               track_id + common_data.random_shortname())
    #                     return False, 0
    #                 elif pred_face == 2:
    #                     common_data.save_face_pic('post', img_big_face, track_id + common_data.random_shortname())
    #                     return True, float(50 * (mouth_score + angle_score))
    #                 elif pred_face == 3:
    #                     common_data.save_face_pic('no_face', img_big_face, track_id + common_data.random_shortname())
    #                     return False, 0
    #             else:
    #                 return False, 0
    #         elif mouth_label == 1:  # 没嘴
    #             common_data.save_face_pic('no_mouth', img_big_face, track_id + common_data.random_shortname())
    #             return False, 0
    #     else:
    #         return False, 0

    #新训练模型
    def evaluate_quality(self, tf, track_id=None, landmark=None):
        # 输入是人脸图片
        # 输入是人脸图片
        start = time.time()

        # img_big_face = tf.face_big_mat
        # mouth_label, mouth_score = self.mouth_check(img_face, landmark)
        mouth_label = 0
        mouth_score = 1.0
        print('mouth_check: %.4fms' % ((time.time() - start) * 1000))
        # if mouth_label is not None:
        #     if mouth_label == 0:  # 有嘴
        start = time.time()
        pred_face, angle_score = self.angle_check(img_face, self.clf_face)
        print('angle_check ：%.4fms' % ((time.time() - start) * 1000))
        if pred_face is not None:
            if pred_face == 0:
                cv2.imwrite('post'+str(a)+'.jpg',img_face)
                # common_data.save_face_pic('post', img_face, track_id + common_data.random_shortname())
                return True, float(50 * (mouth_score + angle_score))
            elif pred_face == 1:
                cv2.imwrite('left' + str(a) + '.jpg',img_face)
                # common_data.save_face_pic('angle_left', img_face, track_id + common_data.random_shortname())
                return False, 0
            elif pred_face == 2:
                cv2.imwrite('right' + str(a) + '.jpg',img_face)
                # common_data.save_face_pic('angle_right', img_face,
                # track_id + common_data.random_shortname())
                return False, 0
            else:
                # common_data.save_face_pic('no_face', img_face, track_id + common_data.random_shortname())
                return False, 0
        else:
            return False, 0
# elif mouth_label == 1:  # 没嘴
                # common_data.save_face_pic('no_mouth', img_face, track_id + common_data.random_shortname())
                # return False, 0
        # else:
        #     return False, 0
def get_files(file_dir):
    angle_left = []
    label_angle_left = []
    angle_right = []
    label_angle_right = []
    # blur = []
    # label_blur = []
    front = []
    label_front = []
    # notface = []
    # label_notface = []
    # downface=[]
    # label_downface=[]
    # shade=[]
    # label_shade=[]
    # 左大角度是0，右大角度是1，正脸是2，不是脸是3，脸向下4,遮挡5
    usrful_ext = ['.jpg']
    # for file in os.listdir(file_dir + '/angle_left'):
    #     file_extension = os.path.splitext(file)[1]  # 分离文件名与扩展名，获取扩展名就是.jpg
    #     if file_extension in usrful_ext:
    #         angle_left.append(file_dir + '/angle_left' + '/' + file)
    #         label_angle_left.append(1)  # 添加标签，该类标签为0，此为2分类例子，多类别识别问题自行添加
    # # # 右大角度是1
    # for file in os.listdir(file_dir + '/angle_right'):
    #     file_extension = os.path.splitext(file)[1]  # 分离文件名与扩展名，获取扩展名就是.jpg
    #     if file_extension in usrful_ext:
    #         angle_right.append(file_dir + '/angle_right' + '/' + file)
    #         label_angle_right.append(2)
    # # 正脸是2
    for file in os.listdir(file_dir + '/front'):
        file_extension = os.path.splitext(file)[1]  # 分离文件名与扩展名，获取扩展名就是.jpg
        if file_extension in usrful_ext:
            front.append(file_dir + '/front' + '/' + file)
            label_front.append(0)
    image_list = np.hstack(( front))  # 所有图片列表
    label_list = np.hstack(( label_front))  # 所有标签列表
    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    # 从打乱的temp中再取出list（img和lab）
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list] #转换成int型
    print("图片读取完成,正在提取训练集/测试集……")
    return image_list, label_list

if __name__ == '__main__':
    # 使用Dlib的正面人脸检测器frontal_face_detector
    detector = dlib.get_frontal_face_detector()
    # dlib的68点模型
    # predictor_path = r'models/landmarks.dat'
    test_dir=r'C:\image\10.15\2018_10_12\1000'
    # # 载入模型
    # clf_mouse_path = r"models/check_mouth.m"
    # clf_face_path = r'models/check_angle.m'
    quality_check1 = FaceQualityCheck()
    # quality_check1.load_model(predictor_path, clf_mouse_path, clf_face_path)
    print('正在启动摄像头……')
    cap = cv2.VideoCapture(0)
    a=0
    # e, img = cap.read()
    image_list, label_list = get_files(test_dir)
    for i in range(0, len(image_list)):  # 测试集
        img = cv2.imread(image_list[i])
        a=a+1

        img_face = face_vector(img)
        if img_face is not None:
            total_time_start = time.time()
            res, score = quality_check1.evaluate_quality(img_face, 1)
            print(score, '       total time %.4fms' % ((time.time() - total_time_start) * 1000))

