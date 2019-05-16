# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from ui_process import Proc_Window

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = Proc_Window()
    mainWindow.show()
    sys.exit(app.exec_())