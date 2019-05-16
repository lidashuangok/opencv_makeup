# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
import cv2 as cv
import numpy as np
#import skin_detector


face_cascade=cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

def big_eye(im,i):
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    eye_rec = eye_cascade.detectMultiScale(gray, 1.3, 5, 0, (30, 30))
    for (x, y, w, h) in eye_rec:
        #im = cv.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #eye = im[int(y+10):int(y + h-15), int(x+4):int(x + w-4)]
        eye = im[int(y):int(y + h ), int(x):int(x + w )]

        #cv.imshow('eye',eye)
        cols, rows,chael = eye.shape
        M = cv.getRotationMatrix2D((w/2, h/2), 0, i)
        eye_after = cv.warpAffine(eye, M, (rows, cols))
        src_mask = np.zeros(eye.shape, eye.dtype)
        # src_mask = 255 * np.ones(tie.shape,tie.dtype)
        poly = np.array([[0,w], [h, w], [h, 0], [0, 0]], np.int32)
        cv.fillPoly(src_mask, [poly], (255, 255, 255))
        center = (int(x + w / 2), int(y + h / 2))
        im = cv.seamlessClone(eye_after, im, src_mask, center, cv.NORMAL_CLONE)
        # cv.imshow('result', result)
        #eye_after = cv.medianBlur(eye_after, 3)
        #im[int(y+10):int(y + h-15), int(x+4):int(x + w-4)]=eye_after
        #im[int(y+10):int(y + h), int(x-3):int(x + w+3)] = eye_after
        #im = cv.GaussianBlur(im, (1,1),0)
    return im
        # pts1 = np.float32([[x, y], [x + w, y + h], [x, y+h]])
        # pts2 = np.float32([[x-i, y-i], [x + w+i, y + h+i], [x, y+h+i]])
        # AffineMatrix = cv.getAffineTransform(np.array(pts1),
        #                                       np.array(pts2))
        # AffineImg = cv.warpAffine(im, AffineMatrix, (im.shape[1], im.shape[0]))
    #cv.imshow('AffineImg',AffineImg)

def face_mop(im,i):
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    faces_rec = face_cascade.detectMultiScale(gray, 1.1, 3)
    for (x,y,w,h) in faces_rec:
        #im = cv.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        #facce = im[int(x):int(x+w),int(y):int(y+h)]
        facce = im[int(y):int(y + h), int(x):int(x + w )]
        #print(facce.shape)
        facce = cv.bilateralFilter(facce, i,30,30)
        #facce = mop(facce)
        #cv.imshow('fa',facce)
        im[int(y):int(y + h), int(x ):int(x + w )] = facce
        #cv.imshow('face',facce)
        #mop(im)
    return im

def meibai(im,i):
    mask = skin_dect(im)
    # mask[int(y):int(y + h), int(x):int(x + w )] =np.ones([h,w,3])
    # cv.imshow('mask',mask)
    # print(mask)
    dst = cv.addWeighted(im, 1, mask, i, 0)
    return dst

def skin_dect(im):
    ycrcb = cv.cvtColor(im, cv.COLOR_BGR2YCrCb)  # 把图像转换到YUV色域
    (y, cr, cb) = cv.split(ycrcb)  # 图像分割, 分别获取y, cr, br通道图像
    # 高斯滤波, cr 是待滤波的源图像数据, (5,5)是值窗口大小, 0 是指根据窗口大小来计算高斯函数标准差
    cr1 = cv.GaussianBlur(cr, (3, 3), 0)  # 对cr通道分量进行高斯滤波
    # 根据OTSU算法求图像阈值, 对图像进行二值化
    _, skin1 = cv.threshold(cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #_, skin1 = cv.threshold(cr1, 139, 255, cv.THRESH_BINARY )
    #print(_)
    mask = cv.cvtColor(skin1,cv.COLOR_GRAY2BGR)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=15)
    #mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel, iterations=1)
    return mask

def shoulian(im):
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    faces_rec = face_cascade.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces_rec:
        # im = cv.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        # facce = im[int(x):int(x+w),int(y):int(y+h)]
        facce = im[int(y+h/2):int(y + h), int(x):int(x + w)]
        cv.imshow('face', facce)
        cols, rows, chael = facce.shape
        M = cv.getRotationMatrix2D((w / 2, h *3/4), 0, 0.9)
        face_after = cv.warpAffine(facce, M, (rows, cols))
        cv.imshow('face_after', face_after)
        # facce = cv.bilateralFilter(facce, i, 30, 30)
        # # facce = mop(facce)
        # # cv.imshow('fa',facce)
        # im[int(y):int(y + h), int(x):int(x + w)] = facce
        # cv.imshow('face',facce)
        # mop(im)
    #return im


if __name__ == '__main__':
    pic = 'test.png'
    im = cv.imread(pic)
    # print(len(eye_rec))
    i=1.3
    #cv.imshow('im', im)
    #result =meibai(im,i) #i=0.1
    result =big_eye(im,i)#1.1
    #shoulian(im)
    #result = face_mop(im, i)#12
    cv.imshow('result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
    #detect(pic)
