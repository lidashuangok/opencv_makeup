# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
from face import Ui_Dialog
from PyQt5 import QtCore,QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import cv2 as cv
import numpy as np
face_cascade=cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
class Proc_Window(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super(Proc_Window, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.loadFile)
        self.meibai.valueChanged.connect(self.meibai_f)
        self.dayan.valueChanged.connect(self.big_eye)
        self.mopi.valueChanged.connect(self.face_mop)

    def loadFile(self):  ########载入file
        try:
            filter = "Images (*.bmp *.png *.jpg)"
            self.image_path, _ = QFileDialog.getOpenFileName(
                self, 'Open image', 'Desktop', filter)
        except:
            print("文件打开错误")

        if self.image_path:
            # global image_pil
            # image_pil = Image.open(image_path)
            # pil读入图片
            # global cv_img_rgb
            self.cv_img_bgr = cv.imread(self.image_path)

            #self.groupBox_slider.setEnabled(True)
            #self.groupBox_button.setEnabled(True)
            # opencv读入图片
            # global image_sk
            # image_sk = io.imread(image_path, as_grey=True)
            # skimage读入图片
            #self.cv_img_bgr_1 = self.cv_img_bgr
            #self.img_before =self.cv_img_bgr
            self.img_before = cv.cvtColor(self.cv_img_bgr, cv.COLOR_BGR2RGB)

            #$self.img_after = self.img_before
            height, width, channel = self.img_before.shape
            bytesPerLine = 3 * width
            self.label.setGeometry(QtCore.QRect(290, 140, 450 * width / height, 450))
            #self.label_pic_after.setGeometry(QtCore.QRect(420, 100, 300 * width / height, 300))
            self.q_image_before = QtGui.QImage(self.img_before.data, width,
                                               height, bytesPerLine, QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(self.q_image_before))

    def big_eye(self):
        i = self.dayan.value()/100
        #print(i)
        gray = cv.cvtColor(self.cv_img_bgr, cv.COLOR_BGR2GRAY)
        eye_rec = eye_cascade.detectMultiScale(gray, 1.3, 5, 0, (30, 30))
        #print(len(eye_rec))

        #for (x, y, w, h) in eye_rec:
                #self.img_before = cv.rectangle(self.img_before, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # eye = im[int(y+10):int(y + h-15), int(x+4):int(x + w-4)]
        x, y, w, h= eye_rec[0]
        eye =  self.cv_img_bgr[int(y):int(y + h), int(x):int(x + w+5)]
        # cv.imshow('eye',eye)
        cols, rows, chael = eye.shape
        M = cv.getRotationMatrix2D((w / 2, h / 2), 0, i)
        eye_after = cv.warpAffine(eye, M, (rows, cols))
        src_mask = np.zeros(eye.shape, eye.dtype)
        # src_mask = 255 * np.ones(tie.shape,tie.dtype)
        poly = np.array([[0, w], [h, w], [h, 0], [0, 0]], np.int32)
        cv.fillPoly(src_mask, [poly], (255, 255, 255))
        center = (int(x + w / 2), int(y + h / 2))
        self.img_before_in = cv.seamlessClone(eye_after,self.cv_img_bgr, src_mask, center, cv.NORMAL_CLONE)
        x, y, w, h = eye_rec[1]
        eye = self.cv_img_bgr[int(y):int(y + h), int(x):int(x + w + 5)]
        # cv.imshow('eye',eye)
        cols, rows, chael = eye.shape
        M = cv.getRotationMatrix2D((w / 2, h / 2), 0, i)
        eye_after = cv.warpAffine(eye, M, (rows, cols))
        src_mask = np.zeros(eye.shape, eye.dtype)
        # src_mask = 255 * np.ones(tie.shape,tie.dtype)
        poly = np.array([[0, w], [h, w], [h, 0], [0, 0]], np.int32)
        cv.fillPoly(src_mask, [poly], (255, 255, 255))
        center = (int(x + w / 2), int(y + h / 2))
        self.img_before = cv.seamlessClone(eye_after, self.img_before_in, src_mask, center, cv.NORMAL_CLONE)

        self.updateIm()

    def face_mop(self):
        i = int(self.mopi.value()/10)
        #print(i)
        #self.cv_img_bgr_1= self.cv_img_bgr
        gray = cv.cvtColor(self.cv_img_bgr, cv.COLOR_BGR2GRAY)
        faces_rec = face_cascade.detectMultiScale(gray, 1.1, 3)
        x, y, w, h =faces_rec[0]
        # im = cv.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        # facce = im[int(x):int(x+w),int(y):int(y+h)]
        facce = self.cv_img_bgr[int(y):int(y + h), int(x):int(x + w)]
        # print(facce.shape)
        #cv.imshow('fa',facce)
        facce_1 = cv.bilateralFilter(facce, i, 30, 30)
        #facce = mop(facce)
        # cv.imshow('fa',facce)
        #cv.imshow('face', self.cv_img_bgr)
        #facce_1 =cv.cvtColor(facce_1,cv.COLOR_BGR2RGB)
        self.img_before[int(y):int(y + h), int(x):int(x + w)] = facce_1
        #cv.imshow('face',self.cv_img_bgr)
        # mop(im)
        self.updateIm()

    def skin_dect(self,im):
        ycrcb = cv.cvtColor(im, cv.COLOR_BGR2YCrCb)  # 把图像转换到YUV色域
        (y, cr, cb) = cv.split(ycrcb)  # 图像分割, 分别获取y, cr, br通道图像
        # 高斯滤波, cr 是待滤波的源图像数据, (5,5)是值窗口大小, 0 是指根据窗口大小来计算高斯函数标准差
        cr1 = cv.GaussianBlur(cr, (3, 3), 0)  # 对cr通道分量进行高斯滤波
        # 根据OTSU算法求图像阈值, 对图像进行二值化
        _, skin1 = cv.threshold(cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # _, skin1 = cv.threshold(cr1, 139, 255, cv.THRESH_BINARY )
        # print(_)
        mask = cv.cvtColor(skin1, cv.COLOR_GRAY2BGR)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=15)
        # mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel, iterations=1)
        return mask

    def meibai_f(self):
        i=self.meibai.value()/1000
        mask = self.skin_dect(self.cv_img_bgr)
        # mask[int(y):int(y + h), int(x):int(x + w )] =np.ones([h,w,3])
        # cv.imshow('mask',mask)
        # print(mask)
        self.img_before = cv.addWeighted(self.cv_img_bgr, 1, mask, i, 0)
        #return dst
        self.updateIm()

    def updateIm(self):
            self.img_before_r = cv.cvtColor(self.img_before,cv.COLOR_BGR2RGB)
            height, width, channel = self.img_before.shape
            bytesPerLine = 3 * width
            #data = self.img_before.tobytes("raw", "RGB")
            self.q_image_after = QtGui.QImage(self.img_before_r, width,height, bytesPerLine, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(self.q_image_after)
            self.label.setPixmap(pix)

