# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'face.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1057, 736)
        self.meibai = QtWidgets.QSlider(Dialog)
        self.meibai.setGeometry(QtCore.QRect(200, 640, 160, 22))
        self.meibai.setMinimum(0)
        self.meibai.setMaximum(200)
        self.meibai.setPageStep(2)
        self.meibai.setOrientation(QtCore.Qt.Horizontal)
        self.meibai.setObjectName("meibai")
        self.dayan = QtWidgets.QSlider(Dialog)
        self.dayan.setGeometry(QtCore.QRect(460, 640, 160, 22))
        self.dayan.setMinimum(100)
        self.dayan.setMaximum(150)
        self.dayan.setOrientation(QtCore.Qt.Horizontal)
        self.dayan.setObjectName("dayan")
        self.mopi = QtWidgets.QSlider(Dialog)
        self.mopi.setGeometry(QtCore.QRect(690, 640, 160, 22))
        self.mopi.setMinimum(10)
        self.mopi.setMaximum(300)
        self.mopi.setSingleStep(2)
        self.mopi.setOrientation(QtCore.Qt.Horizontal)
        self.mopi.setObjectName("mopi")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(680, 70, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(300, 70, 341, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(290, 140, 421, 381))
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(240, 670, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(500, 670, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(720, 670, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "打开文件"))
        self.label_2.setText(_translate("Dialog", "美白"))
        self.label_3.setText(_translate("Dialog", "大眼"))
        self.label_4.setText(_translate("Dialog", "磨皮"))


