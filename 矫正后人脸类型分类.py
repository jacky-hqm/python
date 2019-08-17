'''
建立svm模型，对大角度和正脸进行分类
'''
import numpy
import time
import math
# 使用svm来实现人脸识别
from sklearn import svm
from PIL import Image
import cv2
import os
import numpy as np
import dlib
# 查看模型的准确性
from sklearn.metrics import confusion_matrix
# classification_report看分类报告的结果
from sklearn.metrics import classification_report
# 保存svm的模型
from sklearn.externals import joblib
import shutil


# 读取文件 D:\svm\face_samples2
def get_files(file_dir):
    bigangle_left = []
    label_bigangle_left = []
    bigangle_right = []
    label_bigangle_right = []
    # blur = []
    # label_blur = []
    frontal = []
    label_frontal = []
    notface = []
    label_notface = []
    # downface=[]
    # label_downface=[]
    # shade=[]
    # label_shade=[]
    # 左大角度是0，右大角度是1，正脸是2，不是脸是3，脸向下4,遮挡5
    usrful_ext = ['.jpg']
    for file in os.listdir(file_dir + '/bigangle_left'):
        file_extension = os.path.splitext(file)[1]  # 分离文件名与扩展名，获取扩展名就是.jpg
        if file_extension in usrful_ext:
            bigangle_left.append(file_dir + '/bigangle_left' + '/' + file)
            label_bigangle_left.append(0)  # 添加标签，该类标签为0，此为2分类例子，多类别识别问题自行添加
    # 右大角度是1
    for file in os.listdir(file_dir + '/bigangle_right'):
        file_extension = os.path.splitext(file)[1]  # 分离文件名与扩展名，获取扩展名就是.jpg
        if file_extension in usrful_ext:
            bigangle_right.append(file_dir + '/bigangle_right' + '/' + file)
            label_bigangle_right.append(1)
    # 正脸是2
    for file in os.listdir(file_dir + '/frontal'):
        file_extension = os.path.splitext(file)[1]  # 分离文件名与扩展名，获取扩展名就是.jpg
        if file_extension in usrful_ext:
            frontal.append(file_dir + '/frontal' + '/' + file)
            label_frontal.append(2)
    # 不是脸是3
    for file in os.listdir(file_dir + '/notface'):
        file_extension = os.path.splitext(file)[1]  # 分离文件名与扩展名，获取扩展名就是.jpg
        if file_extension in usrful_ext:
            notface.append(file_dir + '/notface' + '/' + file)
            label_notface.append(3)
    #     # 脸向下是4
    # for file in os.listdir(file_dir + '/downface'):
    #     downface.append(file_dir + '/downface' + '/' + file)
    #     label_downface.append(4)
    #     # 不是脸是3
    # for file in os.listdir(file_dir + '/shade'):
    #     shade.append(file_dir + '/shade' + '/' + file)
    #     label_shade.append(5)

    # 把bigangle和frontal合起来组成一个list（img和lab）
    image_list = np.hstack((bigangle_left, bigangle_right, frontal, notface))  # 所有图片列表
    label_list = np.hstack((label_bigangle_left, label_bigangle_right, label_frontal, label_notface))  # 所有标签列表
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
    # 返回两个list 分别为图片文件名及其标签  顺序已被打乱


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


def eye_features(a):
    # dlib预测器

    #detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor(r'C:\Users\hutao\PycharmProjects\AI\check\vision_work\face_quality\models\landmarks.dat')
     # cv2读取图像


    img=cv2.imread(a)
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 人脸数rects
    rects = detector(img_gray, 0)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
        index = [38,45]
        tmp_points=[] # 存放嘴部的特征点
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            #pos = (point[0, 0], point[0, 1])
            if idx in index:
                pos = (point[0, 0], point[0, 1])  # 每个特征点的x和y
                tmp_points.append(pos)

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
    dist = Distance(eye_left, eye_right)
     # 原图像绕着左眼的坐标旋转。
    image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
    return image


# 训练集提取hog特征
def train(image_list, label_list):
    # 存放训练集图片的列表
    Train_image = []
    Train_label = []

    for i in range(len(image_list)):
        #start1 = time.time()
        pp = eye_features(image_list[i])

        #print(pp)
        if pp is not None:
            leftx = pp[0][0]
            lefty = pp[0][1]
            rightx = pp[1][0]
            righty = pp[1][1]
            image11 = Image.open(image_list[i])
            image=CropFace(image11, eye_left=(leftx, lefty), eye_right=(rightx, righty))
            img1 = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_BGR2GRAY)
        else:

            img1 = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)  # 读取图片，第二个参数表示以灰度图像读入



        if img1 is None:
            print('read failed', image_list[i])
            continue
        res1 = cv2.resize(img1, (64, 64))  # 对图片进行缩放，第一个参数是读入的图片，第二个是制定的缩放大小
        res1 = hog_extract(res1)
        res1_1 = res1.reshape(len(res1) * len(res1[0]))  # 将表示图片的二维矩阵转换成一维
        Train_image.append(res1_1)  # 将list添加到已有的list中
        Train_label.append(label_list[i])

       # print(time.time() - start1)

    Train_image = np.array(Train_image)  # 转化成数组
    Train_label = np.array(Train_label)
    print("训练集提取完成……")
    return Train_image, Train_label


# 测试集
def tes(image_list, label_list):
    # 测试集占20%是642
    Test_image = []
    Test_label = []
    for i in range(0, len(image_list)):  # 测试集


        #start1 = time.time()
        pp = eye_features(image_list[i])

        #print(pp)
        if pp is not None:
            leftx = pp[0][0]
            lefty = pp[0][1]
            rightx = pp[1][0]
            righty = pp[1][1]
            image11 = Image.open(image_list[i])
            image=CropFace(image11, eye_left=(leftx, lefty), eye_right=(rightx, righty))
            img1 = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_BGR2GRAY)
        else:

            img1 = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)  # 读取图片，第二个参数表示以灰度图像读入

        #img1 = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)  # 读取图片，第二个参数表示以灰度图像读入

        if img1 is None:
            print('read failed', image_list[i])
            continue
        res1 = cv2.resize(img1, (64, 64))  # 对图片进行缩放，第一个参数是读入的图片，第二个是制定的缩放大小
        res1 = hog_extract(res1)
        res1_1 = res1.reshape(len(res1) * len(res1[0]))  # 将表示图片的二维矩阵转换成一维
        Test_image.append(res1_1)  # 将list添加到已有的list中
        Test_label.append(label_list[i])
    Test_image = np.array(Test_image)  # 转化成数组
    Test_label = np.array(Test_label)
    print("测试集提取完成，正在计算读取图片时间……")
    return Test_image, Test_label


def model_trian(Train_image, Train_label):
    # 训练模型(如果已经训练，可以直接载入)
    print('正在训练模型')
    clf = svm.SVC(kernel='linear', class_weight='balanced', probability=True)
    clf.fit(Train_image, Train_label)
    # 保存svm模型
    # 设置保存路径
    os.chdir("model_save")
    joblib.dump(clf, "model.m")
    # print('模型训练完成,正在打印训练集分类报告')
    # print("Predict test dataset ...")
    # start = time.clock()
    # y_pred = clf.predict(Test_image)  # 对测试集的预测
    # print("总共耗时Done in {0:.2f}.\n".format(time.clock() - start))
    # # 建立混淆矩阵
    # cm = confusion_matrix(Test_label, y_pred)
    # # 数据中1的个数为a，预测1的次数为b，预测1命中的次数为c
    # # 准确率 precision = c / b
    # # 召回率 recall = c / a
    # # f1_score = 2 * precision * recall / (precision + recall)
    # print("confusion matrix:")
    # np.set_printoptions(threshold=np.nan)
    # print(cm)
    # print(classification_report(Test_label, y_pred))
    # print('预测完成')


# 计算时间
def cal_time(image_list):
    # 载入模型
    clf1 = joblib.load(r"C:\Users\hutao\Desktop\model_save\model.m")

    print('开始计算时间')
    total_time = 0.0
    for i in range(0, int(len(image_list) * 0.1)):  # 测试集取十分之一的样本
        img1 = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)  # 读取图片，第二个参数表示以灰度图像读入
        if img1 is None:
            print('read failed', image_list[i])
            continue
        start1 = time.time()
        res1 = cv2.resize(img1, (64, 64))  # 对图片进行缩放，第一个参数是读入的图片，第二个是制定的缩放大小
        res1 = hog_extract(res1)
        res1_1 = res1.reshape(len(res1) * len(res1[0]))  # 将表示图片的二维矩阵转换成一维
        clf1.predict([res1_1])
        single_time = (time.time() - start1) * 1000
        # print(single_time)
        total_time += single_time
    print('每张图片耗时：{0:.2f}ms'.format((total_time) / int(len(image_list) * 0.1)))
    print('时间计算完成，正在预测并打印分类报告……')


# 预测
def predict(Test_image, Test_label, image_list, flag):
    # 载入模型
    clf = joblib.load(r"C:\Users\hutao\Desktop\model_save\model.m")
    print("Predict test dataset ...")
    start = time.clock()
    y_pred = clf.predict(Test_image)  # 对测试集的预测
    print("总共耗时Done in {0:.2f}.\n".format(time.clock() - start))
    # 建立混淆矩阵
    cm = confusion_matrix(Test_label, y_pred)
    # # 设置分类错误的路径
    # wro_bigangle_left_path = r'\\LT-CHENXIN\yunshike\face_samples\class_wrong\bigangle_left'
    # wro_bigangle_right_path = r'\\LT-CHENXIN\yunshike\face_samples\class_wrong\bigangle_right'
    # wro_frontal_path = r'\\LT-CHENXIN\yunshike\face_samples\class_wrong\frontal'
    # wro_notface_left_path = r'\\LT-CHENXIN\yunshike\face_samples\class_wrong\notface'
    #
    # # fileen=image_list[:\]
    # mark = {'0': 'left', '1': 'right', '2': 'frontal', '3': 'notface'}
    # print('正在保存错误图片……')
    # for i in range(len(Test_label)):
    #     dirname, filename = os.path.split(image_list[i])
    #     if (y_pred[i] != Test_label[i]):
    #         if ((Test_label[i] == 1 or Test_label[i] == 2 or Test_label[i] == 3) and y_pred[i] == 0):
    #             # filename=原来类别_预测类别_第几张
    #             # filename = flag +'_'+ str(Test_label[i]) + '_0_' + str(i) + '.jpg'
    #             filename = flag + '__' + mark[str(Test_label[i])] + '__left__' + filename
    #             shutil.copy(image_list[i], wro_bigangle_left_path + '\\' + filename)
    #
    #         elif ((Test_label[i] == 0 or Test_label[i] == 2 or Test_label[i] == 3) and y_pred[i] == 1):
    #             # filename = flag + str(Test_label[i]) + '_1_' + str(i) + '.jpg'
    #             filename = flag + '__' + mark[str(Test_label[i])] + '__right__' + filename
    #             shutil.copy(image_list[i], wro_bigangle_right_path + '\\' + filename)
    #
    #         elif ((Test_label[i] == 0 or Test_label[i] == 1 or Test_label[i] == 3) and y_pred[i] == 2):
    #             # filename = flag + str(Test_label[i]) + '_2_' + str(i) + '.jpg'
    #             filename = flag + '__' + mark[str(Test_label[i])] + '__frontal__' + filename
    #             shutil.copy(image_list[i], wro_frontal_path + '\\' + filename)
    #
    #         elif ((Test_label[i] == 0 or Test_label[i] == 1 or Test_label[i] == 2) and y_pred[i] == 3):
    #             # filename = flag + str(Test_label[i]) + '_3_' + str(i) + '.jpg'
    #             filename = flag + '__' + mark[str(Test_label[i])] + '__notface__' + filename
    #             shutil.copy(image_list[i], wro_notface_left_path + '\\' + filename)
    #
    # print("保存完成")
    # 数据中1的个数为a，预测1的次数为b，预测1命中的次数为c
    # 准确率 precision = c / b
    # 召回率 recall = c / a
    # f1_score = 2 * precision * recall / (precision + recall)
    print("confusion matrix:")
    np.set_printoptions(threshold=np.nan)
    print(cm)
    print(classification_report(Test_label, y_pred))
    print('预测完成')


if __name__ == '__main__':
    # 设置文件路径
    train_dir = r'C:\Users\hutao\Desktop\train'
    # train_dir = r'D:\svm\test'
    test_dir = r'C:\Users\hutao\Desktop\test'
    # 获取图片和标签列表
    image_list, label_list = get_files(train_dir)
    print(len(image_list))
    print(len(label_list))
    #########################################################################

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r'C:\Users\hutao\PycharmProjects\AI\check\vision_work\face_quality\models\landmarks.dat')

    # 以下为训练
    # 存放训练集图片的列表

    Train_image, Train_label = train(image_list, label_list)
    # 训练模型并保存
    model_trian(Train_image, Train_label)
    # # 计算时间
    cal_time(image_list)
    # # 预测
    predict(Train_image, Train_label, image_list, flag='Train')

    #########################################################################
    #以下为测试
    # 获取图片和标签列表
    image_list, label_list = get_files(test_dir)
    print(len(image_list))
    print(len(label_list))
    #
    # # 存放测试集图片的列表
    Test_image, Test_label = tes(image_list, label_list)
    # # 计算时间
    cal_time(image_list)
    # # 预测
    predict(Test_image, Test_label, image_list, flag='Test')
