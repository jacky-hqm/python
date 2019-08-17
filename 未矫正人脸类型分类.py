'''
建立svm模型，对大角度和正脸进行分类
'''
import time
# 使用svm来实现人脸识别
from sklearn import svm
import cv2
import os
import numpy as np
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


# 训练集提取hog特征
def train(image_list, label_list):
    # 存放训练集图片的列表
    Train_image = []
    Train_label = []
    for i in range(len(image_list)):
        img1 = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)  # 读取图片，第二个参数表示以灰度图像读入
        if img1 is None:
            print('read failed', image_list[i])
            continue
        res1 = cv2.resize(img1, (64, 64))  # 对图片进行缩放，第一个参数是读入的图片，第二个是制定的缩放大小
        res1 = hog_extract(res1)
        res1_1 = res1.reshape(len(res1) * len(res1[0]))  # 将表示图片的二维矩阵转换成一维
        Train_image.append(res1_1)  # 将list添加到已有的list中
        Train_label.append(label_list[i])
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
        img1 = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)  # 读取图片，第二个参数表示以灰度图像读入
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
