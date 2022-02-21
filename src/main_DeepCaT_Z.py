# -*- coding: utf-8 -*-
"""
@author: DeepCaT_Z
"""

from PyQt5 import QtCore, QtGui, QtWidgets

# **************** Imports ***************************************

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication, QTimer, QDateTime
import pyrealsense2 as rs

from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap, QCursor

import numpy as np
import cv2
import keyboard
import serial
import serial.tools.list_ports
import time
import os
from skimage.transform import resize
from skimage.io import imread, imsave

import torch

from config_DeepCaT_Z import *
import classificationModel_functions
from segmentationModel_functions import UNet_ConvLSTM, UNet_simple

import tensorflow as tf

import imageio
import pandas as pd
from datetime import date

# ******************** MAIN CLASS - DO NOT TOUCH **************************

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1162, 807)
        MainWindow.setMinimumSize(QtCore.QSize(1162, 807))
        MainWindow.setMaximumSize(QtCore.QSize(1162, 807))
        MainWindow.setStyleSheet("background-color: rgb(59, 59, 59);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.setup_gb = QtWidgets.QGroupBox(self.centralwidget)
        self.setup_gb.setGeometry(QtCore.QRect(10, 10, 441, 781))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.setup_gb.setFont(font)
        self.setup_gb.setStyleSheet("background-color: rgb(0, 68, 100);\n"
"border-radius:10px;")
        self.setup_gb.setTitle("")
        self.setup_gb.setObjectName("setup_gb")
        self.setup_lbl = QtWidgets.QLabel(self.setup_gb)
        self.setup_lbl.setGeometry(QtCore.QRect(10, 10, 61, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.setup_lbl.setFont(font)
        self.setup_lbl.setStyleSheet("\n"
"color: rgb(255, 255, 255);\n"
"")
        self.setup_lbl.setObjectName("setup_lbl")
        self.camera_gb = QtWidgets.QGroupBox(self.setup_gb)
        self.camera_gb.setGeometry(QtCore.QRect(19, 34, 171, 231))
        self.camera_gb.setStyleSheet("background-color: rgb(0, 90, 132);\n"
"border-radius:10px;")
        self.camera_gb.setTitle("")
        self.camera_gb.setObjectName("camera_gb")
        self.cam_connection_lbl = QtWidgets.QLabel(self.camera_gb)
        self.cam_connection_lbl.setGeometry(QtCore.QRect(0, 6, 171, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.cam_connection_lbl.setFont(font)
        self.cam_connection_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.cam_connection_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.cam_connection_lbl.setObjectName("cam_connection_lbl")
        self.cam_connect_pb = QtWidgets.QPushButton(self.camera_gb)
        self.cam_connect_pb.setGeometry(QtCore.QRect(43, 33, 81, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.cam_connect_pb.setFont(font)
        self.cam_connect_pb.setStyleSheet("color: rgb(0, 0, 0);\n"
"border-radius:10px;background-color: rgb(195, 195, 195);\n"
"")
        self.cam_connect_pb.setObjectName("cam_connect_pb")
        self.warning_cam_connect_lbl = QtWidgets.QLabel(self.camera_gb)
        self.warning_cam_connect_lbl.setGeometry(QtCore.QRect(6, 60, 161, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.warning_cam_connect_lbl.setFont(font)
        self.warning_cam_connect_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.warning_cam_connect_lbl.setText("")
        self.warning_cam_connect_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.warning_cam_connect_lbl.setWordWrap(True)
        self.warning_cam_connect_lbl.setObjectName("warning_cam_connect_lbl")
        self.cam_showpreview_pb = QtWidgets.QPushButton(self.camera_gb)
        self.cam_showpreview_pb.setEnabled(False)
        self.cam_showpreview_pb.setGeometry(QtCore.QRect(30, 125, 111, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.cam_showpreview_pb.setFont(font)
        self.cam_showpreview_pb.setStyleSheet("color: rgb(0, 0, 0);\n"
"border-radius:10px;\n"
"background-color: rgb(203, 203, 203);\n"
"")
        self.cam_showpreview_pb.setCheckable(False)
        self.cam_showpreview_pb.setAutoDefault(False)
        self.cam_showpreview_pb.setDefault(False)
        self.cam_showpreview_pb.setFlat(False)
        self.cam_showpreview_pb.setObjectName("cam_showpreview_pb")
        self.cam_takesnapshot_pb = QtWidgets.QPushButton(self.camera_gb)
        self.cam_takesnapshot_pb.setEnabled(False)
        self.cam_takesnapshot_pb.setGeometry(QtCore.QRect(30, 178, 111, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.cam_takesnapshot_pb.setFont(font)
        self.cam_takesnapshot_pb.setStyleSheet("color: rgb(0, 0, 0);\n"
"border-radius:10px;\n"
"background-color: rgb(203, 203, 203);\n"
"")
        self.cam_takesnapshot_pb.setCheckable(False)
        self.cam_takesnapshot_pb.setAutoDefault(False)
        self.cam_takesnapshot_pb.setDefault(False)
        self.cam_takesnapshot_pb.setFlat(False)
        self.cam_takesnapshot_pb.setObjectName("cam_takesnapshot_pb")
        self.arduino_gb = QtWidgets.QGroupBox(self.setup_gb)
        self.arduino_gb.setGeometry(QtCore.QRect(210, 34, 211, 101))
        self.arduino_gb.setStyleSheet("background-color: rgb(89, 90, 95);\n"
"border-radius:10px;")
        self.arduino_gb.setTitle("")
        self.arduino_gb.setObjectName("arduino_gb")
        self.ard_connection_lbl = QtWidgets.QLabel(self.arduino_gb)
        self.ard_connection_lbl.setGeometry(QtCore.QRect(0, 6, 211, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.ard_connection_lbl.setFont(font)
        self.ard_connection_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.ard_connection_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.ard_connection_lbl.setObjectName("ard_connection_lbl")
        self.ard_connect_pb = QtWidgets.QPushButton(self.arduino_gb)
        self.ard_connect_pb.setGeometry(QtCore.QRect(67, 30, 81, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.ard_connect_pb.setFont(font)
        self.ard_connect_pb.setStyleSheet("color: rgb(0, 0, 0);\n"
"border-radius:10px;\n"
"background-color: rgb(195, 195, 195);\n"
"")
        self.ard_connect_pb.setObjectName("ard_connect_pb")
        self.warning_ard_connect_lbl = QtWidgets.QLabel(self.arduino_gb)
        self.warning_ard_connect_lbl.setGeometry(QtCore.QRect(24, 60, 161, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.warning_ard_connect_lbl.setFont(font)
        self.warning_ard_connect_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.warning_ard_connect_lbl.setText("")
        self.warning_ard_connect_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.warning_ard_connect_lbl.setWordWrap(True)
        self.warning_ard_connect_lbl.setObjectName("warning_ard_connect_lbl")
        self.ard_connection_lbl_2 = QtWidgets.QLabel(self.arduino_gb)
        self.ard_connection_lbl_2.setGeometry(QtCore.QRect(180, 110, 181, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.ard_connection_lbl_2.setFont(font)
        self.ard_connection_lbl_2.setStyleSheet("color: rgb(0, 0, 0);")
        self.ard_connection_lbl_2.setAlignment(QtCore.Qt.AlignCenter)
        self.ard_connection_lbl_2.setObjectName("ard_connection_lbl_2")
        self.deeplearning_gb = QtWidgets.QGroupBox(self.setup_gb)
        self.deeplearning_gb.setGeometry(QtCore.QRect(20, 284, 401, 361))
        self.deeplearning_gb.setStyleSheet("background-color: rgb(126, 126, 126);\n"
"border-radius:10px;")
        self.deeplearning_gb.setTitle("")
        self.deeplearning_gb.setObjectName("deeplearning_gb")
        self.deep_lbl = QtWidgets.QLabel(self.deeplearning_gb)
        self.deep_lbl.setGeometry(QtCore.QRect(0, 11, 401, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.deep_lbl.setFont(font)
        self.deep_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.deep_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.deep_lbl.setObjectName("deep_lbl")
        self.choose_behav_pb = QtWidgets.QPushButton(self.deeplearning_gb)
        self.choose_behav_pb.setGeometry(QtCore.QRect(150, 49, 81, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.choose_behav_pb.setFont(font)
        self.choose_behav_pb.setStyleSheet("color: rgb(0, 0, 0);\n"
"border-radius:10px;\n"
"background-color: rgb(195, 195, 195);\n"
"")
        self.choose_behav_pb.setObjectName("choose_behav_pb")
        self.classific_lbl = QtWidgets.QLabel(self.deeplearning_gb)
        self.classific_lbl.setGeometry(QtCore.QRect(20, 49, 121, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.classific_lbl.setFont(font)
        self.classific_lbl.setStyleSheet("color: rgb(0, 0, 0);")
        self.classific_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.classific_lbl.setObjectName("classific_lbl")
        self.segment_lbl = QtWidgets.QLabel(self.deeplearning_gb)
        self.segment_lbl.setGeometry(QtCore.QRect(20, 89, 121, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.segment_lbl.setFont(font)
        self.segment_lbl.setStyleSheet("color: rgb(0, 0, 0);")
        self.segment_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.segment_lbl.setObjectName("segment_lbl")
        self.warning_deepbehav_lbl = QtWidgets.QLabel(self.deeplearning_gb)
        self.warning_deepbehav_lbl.setGeometry(QtCore.QRect(240, 48, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.warning_deepbehav_lbl.setFont(font)
        self.warning_deepbehav_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.warning_deepbehav_lbl.setText("")
        self.warning_deepbehav_lbl.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.warning_deepbehav_lbl.setWordWrap(True)
        self.warning_deepbehav_lbl.setObjectName("warning_deepbehav_lbl")
        self.warning_segmodels_lbl = QtWidgets.QLabel(self.deeplearning_gb)
        self.warning_segmodels_lbl.setGeometry(QtCore.QRect(235, 155, 141, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.warning_segmodels_lbl.setFont(font)
        self.warning_segmodels_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.warning_segmodels_lbl.setText("")
        self.warning_segmodels_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.warning_segmodels_lbl.setWordWrap(True)
        self.warning_segmodels_lbl.setObjectName("warning_segmodels_lbl")
        self.segparam_gb = QtWidgets.QGroupBox(self.deeplearning_gb)
        self.segparam_gb.setGeometry(QtCore.QRect(0, 240, 401, 121))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.segparam_gb.setFont(font)
        self.segparam_gb.setStyleSheet("color: rgb(62, 74, 84);\n"
"background-color: rgb(167, 167, 167);")
        self.segparam_gb.setTitle("")
        self.segparam_gb.setObjectName("segparam_gb")
        self.segment_lbl_2 = QtWidgets.QLabel(self.segparam_gb)
        self.segment_lbl_2.setGeometry(QtCore.QRect(-10, 0, 421, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.segment_lbl_2.setFont(font)
        self.segment_lbl_2.setStyleSheet("color: rgb(195, 195, 195);\n"
"background-color: rgb(89, 90, 95);")
        self.segment_lbl_2.setAlignment(QtCore.Qt.AlignCenter)
        self.segment_lbl_2.setObjectName("segment_lbl_2")
        self.Y_border_sb = QtWidgets.QSpinBox(self.segparam_gb)
        self.Y_border_sb.setGeometry(QtCore.QRect(233, 30, 41, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.Y_border_sb.setFont(font)
        self.Y_border_sb.setStyleSheet("background-color: rgb(230, 230, 230);")
        self.Y_border_sb.setMinimum(0)
        self.Y_border_sb.setMaximum(150)
        self.Y_border_sb.setSingleStep(1)
        self.Y_border_sb.setProperty("value", 40)
        self.Y_border_sb.setObjectName("Y_border_sb")
        self.X_lbl = QtWidgets.QLabel(self.segparam_gb)
        self.X_lbl.setGeometry(QtCore.QRect(96, 33, 31, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        self.X_lbl.setFont(font)
        self.X_lbl.setStyleSheet("color: rgb(83, 83, 83);")
        self.X_lbl.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.X_lbl.setWordWrap(True)
        self.X_lbl.setObjectName("X_lbl")
        self.border_lbl = QtWidgets.QLabel(self.segparam_gb)
        self.border_lbl.setGeometry(QtCore.QRect(11, 30, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.border_lbl.setFont(font)
        self.border_lbl.setStyleSheet("color: rgb(83, 83, 83);")
        self.border_lbl.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.border_lbl.setWordWrap(True)
        self.border_lbl.setObjectName("border_lbl")
        self.Y_lbl = QtWidgets.QLabel(self.segparam_gb)
        self.Y_lbl.setGeometry(QtCore.QRect(193, 33, 41, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        self.Y_lbl.setFont(font)
        self.Y_lbl.setStyleSheet("color: rgb(83, 83, 83);")
        self.Y_lbl.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.Y_lbl.setWordWrap(True)
        self.Y_lbl.setObjectName("Y_lbl")
        self.X_border_sb = QtWidgets.QSpinBox(self.segparam_gb)
        self.X_border_sb.setGeometry(QtCore.QRect(134, 30, 43, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.X_border_sb.setFont(font)
        self.X_border_sb.setStyleSheet("background-color: rgb(230, 230, 230);")
        self.X_border_sb.setMaximum(150)
        self.X_border_sb.setProperty("value", 100)
        self.X_border_sb.setObjectName("X_border_sb")
        self.MaxRange_sb = QtWidgets.QSpinBox(self.segparam_gb)
        self.MaxRange_sb.setGeometry(QtCore.QRect(304, 70, 43, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.MaxRange_sb.setFont(font)
        self.MaxRange_sb.setStyleSheet("background-color: rgb(230, 230, 230);")
        self.MaxRange_sb.setMinimum(10)
        self.MaxRange_sb.setMaximum(1000)
        self.MaxRange_sb.setSingleStep(10)
        self.MaxRange_sb.setProperty("value", 300)
        self.MaxRange_sb.setObjectName("MaxRange_sb")
        self.min_lbl = QtWidgets.QLabel(self.segparam_gb)
        self.min_lbl.setGeometry(QtCore.QRect(150, 72, 61, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        self.min_lbl.setFont(font)
        self.min_lbl.setStyleSheet("color: rgb(83, 83, 83);")
        self.min_lbl.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.min_lbl.setWordWrap(True)
        self.min_lbl.setObjectName("min_lbl")
        self.max_lbl = QtWidgets.QLabel(self.segparam_gb)
        self.max_lbl.setGeometry(QtCore.QRect(252, 73, 51, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        self.max_lbl.setFont(font)
        self.max_lbl.setStyleSheet("color: rgb(83, 83, 83);")
        self.max_lbl.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.max_lbl.setWordWrap(True)
        self.max_lbl.setObjectName("max_lbl")
        self.MinRange_sb = QtWidgets.QSpinBox(self.segparam_gb)
        self.MinRange_sb.setGeometry(QtCore.QRect(200, 70, 43, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.MinRange_sb.setFont(font)
        self.MinRange_sb.setStyleSheet("background-color: rgb(230, 230, 230);")
        self.MinRange_sb.setMaximum(10)
        self.MinRange_sb.setProperty("value", 5)
        self.MinRange_sb.setObjectName("MinRange_sb")
        self.border_lbl_2 = QtWidgets.QLabel(self.segparam_gb)
        self.border_lbl_2.setGeometry(QtCore.QRect(10, 70, 141, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.border_lbl_2.setFont(font)
        self.border_lbl_2.setStyleSheet("color: rgb(83, 83, 83);")
        self.border_lbl_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.border_lbl_2.setWordWrap(True)
        self.border_lbl_2.setObjectName("border_lbl_2")
        self.choose_segmodel_deep_pb = QtWidgets.QPushButton(self.deeplearning_gb)
        self.choose_segmodel_deep_pb.setGeometry(QtCore.QRect(150, 156, 81, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.choose_segmodel_deep_pb.setFont(font)
        self.choose_segmodel_deep_pb.setStyleSheet("color: rgb(0, 0, 0);\n"
"border-radius:10px;\n"
"background-color: rgb(195, 195, 195);\n"
"")
        self.choose_segmodel_deep_pb.setObjectName("choose_segmodel_deep_pb")
        self.load_segmodel_back_pb = QtWidgets.QPushButton(self.deeplearning_gb)
        self.load_segmodel_back_pb.setGeometry(QtCore.QRect(70, 156, 81, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.load_segmodel_back_pb.setFont(font)
        self.load_segmodel_back_pb.setStyleSheet("color: rgb(0, 0, 0);\n"
"border-radius:10px;\n"
"background-color: rgb(195, 195, 195);\n"
"")
        self.load_segmodel_back_pb.setObjectName("load_segmodel_back_pb")
        self.create_segmodel_back_pb = QtWidgets.QPushButton(self.deeplearning_gb)
        self.create_segmodel_back_pb.setGeometry(QtCore.QRect(230, 156, 81, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.create_segmodel_back_pb.setFont(font)
        self.create_segmodel_back_pb.setStyleSheet("color: rgb(0, 0, 0);\n"
"border-radius:10px;\n"
"background-color: rgb(195, 195, 195);\n"
"")
        self.create_segmodel_back_pb.setObjectName("create_segmodel_back_pb")
        self.warning_create_segmodel_back = QtWidgets.QLabel(self.deeplearning_gb)
        self.warning_create_segmodel_back.setGeometry(QtCore.QRect(190, 190, 161, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.warning_create_segmodel_back.setFont(font)
        self.warning_create_segmodel_back.setStyleSheet("color: rgb(255, 255, 255);")
        self.warning_create_segmodel_back.setText("")
        self.warning_create_segmodel_back.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.warning_create_segmodel_back.setWordWrap(True)
        self.warning_create_segmodel_back.setObjectName("warning_create_segmodel_back")
        self.groupBox = QtWidgets.QGroupBox(self.deeplearning_gb)
        self.groupBox.setGeometry(QtCore.QRect(141, 79, 191, 61))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.backsub_model_rb = QtWidgets.QRadioButton(self.groupBox)
        self.backsub_model_rb.setGeometry(QtCore.QRect(10, 43, 181, 17))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.backsub_model_rb.setFont(font)
        self.backsub_model_rb.setObjectName("backsub_model_rb")
        self.deep_model_rb = QtWidgets.QRadioButton(self.groupBox)
        self.deep_model_rb.setGeometry(QtCore.QRect(10, 13, 171, 17))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.deep_model_rb.setFont(font)
        self.deep_model_rb.setChecked(True)
        self.deep_model_rb.setObjectName("deep_model_rb")
        self.UNet_gb = QtWidgets.QGroupBox(self.deeplearning_gb)
        self.UNet_gb.setGeometry(QtCore.QRect(130, 187, 161, 41))
        self.UNet_gb.setTitle("")
        self.UNet_gb.setObjectName("UNet_gb")
        self.ConvLSTM_rb = QtWidgets.QRadioButton(self.UNet_gb)
        self.ConvLSTM_rb.setGeometry(QtCore.QRect(27, 24, 191, 17))
        self.ConvLSTM_rb.setObjectName("ConvLSTM_rb")
        self.UNet_rb = QtWidgets.QRadioButton(self.UNet_gb)
        self.UNet_rb.setGeometry(QtCore.QRect(27, 1, 82, 17))
        self.UNet_rb.setChecked(True)
        self.UNet_rb.setObjectName("UNet_rb")
        self.warning_load_segmodelback = QtWidgets.QLabel(self.deeplearning_gb)
        self.warning_load_segmodelback.setGeometry(QtCore.QRect(37, 190, 141, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.warning_load_segmodelback.setFont(font)
        self.warning_load_segmodelback.setStyleSheet("color: rgb(255, 255, 255);")
        self.warning_load_segmodelback.setText("")
        self.warning_load_segmodelback.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.warning_load_segmodelback.setWordWrap(True)
        self.warning_load_segmodelback.setObjectName("warning_load_segmodelback")
        self.camera_gb_2 = QtWidgets.QGroupBox(self.setup_gb)
        self.camera_gb_2.setGeometry(QtCore.QRect(20, 664, 401, 91))
        self.camera_gb_2.setStyleSheet("background-color: rgb(105, 118, 102);\n"
"border-radius:10px;")
        self.camera_gb_2.setTitle("")
        self.camera_gb_2.setObjectName("camera_gb_2")
        self.save_param_lbl = QtWidgets.QLabel(self.camera_gb_2)
        self.save_param_lbl.setGeometry(QtCore.QRect(0, 4, 401, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.save_param_lbl.setFont(font)
        self.save_param_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.save_param_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.save_param_lbl.setObjectName("save_param_lbl")
        self.open_folder_pb = QtWidgets.QPushButton(self.camera_gb_2)
        self.open_folder_pb.setGeometry(QtCore.QRect(350, 32, 31, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.open_folder_pb.setFont(font)
        self.open_folder_pb.setStyleSheet("color: rgb(0, 0, 0);\n"
"border-radius:10px;\n"
"background-color: rgb(122, 122, 122);\n"
"")
        self.open_folder_pb.setObjectName("open_folder_pb")
        self.selectfolder_lbl = QtWidgets.QLabel(self.camera_gb_2)
        self.selectfolder_lbl.setGeometry(QtCore.QRect(10, 34, 91, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.selectfolder_lbl.setFont(font)
        self.selectfolder_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.selectfolder_lbl.setObjectName("selectfolder_lbl")
        self.warning_selectfolder_lbl = QtWidgets.QLabel(self.camera_gb_2)
        self.warning_selectfolder_lbl.setGeometry(QtCore.QRect(115, 55, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.warning_selectfolder_lbl.setFont(font)
        self.warning_selectfolder_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.warning_selectfolder_lbl.setText("")
        self.warning_selectfolder_lbl.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.warning_selectfolder_lbl.setWordWrap(True)
        self.warning_selectfolder_lbl.setObjectName("warning_selectfolder_lbl")
        self.directory_edt = QtWidgets.QLineEdit(self.camera_gb_2)
        self.directory_edt.setEnabled(True)
        self.directory_edt.setGeometry(QtCore.QRect(110, 34, 221, 20))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        self.directory_edt.setFont(font)
        self.directory_edt.setStyleSheet("color: rgb(0, 85, 127);\n"
"background-color: rgb(195, 195, 195);")
        self.directory_edt.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.directory_edt.setAlignment(QtCore.Qt.AlignCenter)
        self.directory_edt.setObjectName("directory_edt")
        self.arduino_gb_2 = QtWidgets.QGroupBox(self.setup_gb)
        self.arduino_gb_2.setGeometry(QtCore.QRect(210, 154, 211, 111))
        self.arduino_gb_2.setStyleSheet("background-color: rgb(167, 167, 167);\n"
"border-radius:10px;")
        self.arduino_gb_2.setTitle("")
        self.arduino_gb_2.setObjectName("arduino_gb_2")
        self.ard_connection_lbl_3 = QtWidgets.QLabel(self.arduino_gb_2)
        self.ard_connection_lbl_3.setGeometry(QtCore.QRect(0, 4, 211, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.ard_connection_lbl_3.setFont(font)
        self.ard_connection_lbl_3.setStyleSheet("color: rgb(255, 255, 255);")
        self.ard_connection_lbl_3.setAlignment(QtCore.Qt.AlignCenter)
        self.ard_connection_lbl_3.setObjectName("ard_connection_lbl_3")
        self.ard_connection_lbl_4 = QtWidgets.QLabel(self.arduino_gb_2)
        self.ard_connection_lbl_4.setGeometry(QtCore.QRect(180, 110, 181, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.ard_connection_lbl_4.setFont(font)
        self.ard_connection_lbl_4.setStyleSheet("color: rgb(0, 0, 0);")
        self.ard_connection_lbl_4.setAlignment(QtCore.Qt.AlignCenter)
        self.ard_connection_lbl_4.setObjectName("ard_connection_lbl_4")
        self.deptherror_lbl = QtWidgets.QLabel(self.arduino_gb_2)
        self.deptherror_lbl.setGeometry(QtCore.QRect(8, 38, 131, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.deptherror_lbl.setFont(font)
        self.deptherror_lbl.setToolTip("")
        self.deptherror_lbl.setStyleSheet("color: rgb(83, 83, 83);")
        self.deptherror_lbl.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.deptherror_lbl.setWordWrap(True)
        self.deptherror_lbl.setObjectName("deptherror_lbl")
        self.min_meters_sb = QtWidgets.QSpinBox(self.arduino_gb_2)
        self.min_meters_sb.setGeometry(QtCore.QRect(58, 66, 43, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.min_meters_sb.setFont(font)
        self.min_meters_sb.setStyleSheet("background-color: rgb(230, 230, 230);")
        self.min_meters_sb.setMaximum(2000)
        self.min_meters_sb.setProperty("value", 1024)
        self.min_meters_sb.setObjectName("min_meters_sb")
        self.min_meters_lbl = QtWidgets.QLabel(self.arduino_gb_2)
        self.min_meters_lbl.setGeometry(QtCore.QRect(9, 67, 51, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        self.min_meters_lbl.setFont(font)
        self.min_meters_lbl.setStyleSheet("color: rgb(83, 83, 83);")
        self.min_meters_lbl.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.min_meters_lbl.setWordWrap(True)
        self.min_meters_lbl.setObjectName("min_meters_lbl")
        self.max_meters_sb = QtWidgets.QSpinBox(self.arduino_gb_2)
        self.max_meters_sb.setEnabled(False)
        self.max_meters_sb.setGeometry(QtCore.QRect(163, 66, 43, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.max_meters_sb.setFont(font)
        self.max_meters_sb.setStyleSheet("background-color: rgb(230, 230, 230);")
        self.max_meters_sb.setMinimum(0)
        self.max_meters_sb.setMaximum(2000)
        self.max_meters_sb.setSingleStep(1)
        self.max_meters_sb.setProperty("value", 1279)
        self.max_meters_sb.setObjectName("max_meters_sb")
        self.max_meters_lbl = QtWidgets.QLabel(self.arduino_gb_2)
        self.max_meters_lbl.setGeometry(QtCore.QRect(112, 69, 51, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        self.max_meters_lbl.setFont(font)
        self.max_meters_lbl.setStyleSheet("color: rgb(83, 83, 83);")
        self.max_meters_lbl.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.max_meters_lbl.setWordWrap(True)
        self.max_meters_lbl.setObjectName("max_meters_lbl")
        self.control_gb = QtWidgets.QGroupBox(self.centralwidget)
        self.control_gb.setGeometry(QtCore.QRect(470, 10, 681, 781))
        self.control_gb.setStyleSheet("background-color: rgb(0, 68, 100);\n"
"border-radius:10px;")
        self.control_gb.setTitle("")
        self.control_gb.setObjectName("control_gb")
        self.setup_lbl_3 = QtWidgets.QLabel(self.control_gb)
        self.setup_lbl_3.setGeometry(QtCore.QRect(10, 10, 241, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.setup_lbl_3.setFont(font)
        self.setup_lbl_3.setStyleSheet("color: rgb(255, 255, 255);\n"
"")
        self.setup_lbl_3.setObjectName("setup_lbl_3")
        self.preview_realtime_lbl = QtWidgets.QLabel(self.control_gb)
        self.preview_realtime_lbl.setGeometry(QtCore.QRect(20, 215, 640, 480))
        self.preview_realtime_lbl.setStyleSheet("background-color: rgb(0, 54, 80);")
        self.preview_realtime_lbl.setText("")
        self.preview_realtime_lbl.setObjectName("preview_realtime_lbl")
        self.camera_gb_5 = QtWidgets.QGroupBox(self.control_gb)
        self.camera_gb_5.setGeometry(QtCore.QRect(22, 34, 251, 161))
        self.camera_gb_5.setStyleSheet("background-color: rgb(0, 90, 132);\n"
"border-radius:10px;")
        self.camera_gb_5.setTitle("")
        self.camera_gb_5.setObjectName("camera_gb_5")
        self.territories_lbl = QtWidgets.QLabel(self.camera_gb_5)
        self.territories_lbl.setGeometry(QtCore.QRect(0, 4, 251, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.territories_lbl.setFont(font)
        self.territories_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.territories_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.territories_lbl.setObjectName("territories_lbl")
        self.load_territories_pb = QtWidgets.QPushButton(self.camera_gb_5)
        self.load_territories_pb.setGeometry(QtCore.QRect(34, 32, 91, 23))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.load_territories_pb.setFont(font)
        self.load_territories_pb.setStyleSheet("color: rgb(0, 0, 0);\n"
"border-radius:10px;\n"
"background-color: rgb(195, 195, 195);\n"
"")
        self.load_territories_pb.setObjectName("load_territories_pb")
        self.warning_load_territories_lbl = QtWidgets.QLabel(self.camera_gb_5)
        self.warning_load_territories_lbl.setGeometry(QtCore.QRect(127, 33, 81, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.warning_load_territories_lbl.setFont(font)
        self.warning_load_territories_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.warning_load_territories_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.warning_load_territories_lbl.setObjectName("warning_load_territories_lbl")
        self.territories_preview_lbl = QtWidgets.QLabel(self.camera_gb_5)
        self.territories_preview_lbl.setGeometry(QtCore.QRect(54, 69, 131, 81))
        self.territories_preview_lbl.setStyleSheet("background-color: rgb(0, 54, 80);")
        self.territories_preview_lbl.setText("")
        self.territories_preview_lbl.setObjectName("territories_preview_lbl")
        self.start_pb = QtWidgets.QPushButton(self.control_gb)
        self.start_pb.setGeometry(QtCore.QRect(109, 702, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.start_pb.setFont(font)
        self.start_pb.setStyleSheet("color: rgb(255, 255, 255);\n"
"border-radius:10px;\n"
"background-color: rgb(117, 131, 113);\n"
"")
        self.start_pb.setObjectName("start_pb")
        self.stop_pb = QtWidgets.QPushButton(self.control_gb)
        self.stop_pb.setGeometry(QtCore.QRect(502, 702, 75, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.stop_pb.setFont(font)
        self.stop_pb.setStyleSheet("color: rgb(255, 255, 255);\n"
"border-radius:10px;\n"
"background-color: rgb(85, 0, 0);")
        self.stop_pb.setObjectName("stop_pb")
        self.label_3 = QtWidgets.QLabel(self.control_gb)
        self.label_3.setGeometry(QtCore.QRect(290, 712, 47, 13))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_3.setObjectName("label_3")
        self.directory_edt_2 = QtWidgets.QLineEdit(self.control_gb)
        self.directory_edt_2.setEnabled(False)
        self.directory_edt_2.setGeometry(QtCore.QRect(333, 708, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.directory_edt_2.setFont(font)
        self.directory_edt_2.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(195, 195, 195);")
        self.directory_edt_2.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.directory_edt_2.setAlignment(QtCore.Qt.AlignCenter)
        self.directory_edt_2.setObjectName("directory_edt_2")
        self.environment_gb = QtWidgets.QGroupBox(self.control_gb)
        self.environment_gb.setGeometry(QtCore.QRect(296, 34, 361, 161))
        self.environment_gb.setStyleSheet("background-color: rgb(89, 90, 95);\n"
"border-radius:10px;")
        self.environment_gb.setTitle("")
        self.environment_gb.setObjectName("environment_gb")
        self.environment_lbl = QtWidgets.QLabel(self.environment_gb)
        self.environment_lbl.setGeometry(QtCore.QRect(0, 4, 351, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.environment_lbl.setFont(font)
        self.environment_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.environment_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.environment_lbl.setObjectName("environment_lbl")
        self.sensor_lbl = QtWidgets.QLabel(self.environment_gb)
        self.sensor_lbl.setGeometry(QtCore.QRect(40, 31, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.sensor_lbl.setFont(font)
        self.sensor_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.sensor_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.sensor_lbl.setObjectName("sensor_lbl")
        self.sensor1_edt = QtWidgets.QLineEdit(self.environment_gb)
        self.sensor1_edt.setEnabled(False)
        self.sensor1_edt.setGeometry(QtCore.QRect(66, 52, 21, 22))
        self.sensor1_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
        self.sensor1_edt.setObjectName("sensor1_edt")
        self.sensor2_edt = QtWidgets.QLineEdit(self.environment_gb)
        self.sensor2_edt.setEnabled(False)
        self.sensor2_edt.setGeometry(QtCore.QRect(66, 79, 21, 21))
        self.sensor2_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
        self.sensor2_edt.setObjectName("sensor2_edt")
        self.sensor3_edt = QtWidgets.QLineEdit(self.environment_gb)
        self.sensor3_edt.setEnabled(False)
        self.sensor3_edt.setGeometry(QtCore.QRect(66, 105, 21, 22))
        self.sensor3_edt.setStyleSheet("background-color: rgb(130, 130, 130);")
        self.sensor3_edt.setObjectName("sensor3_edt")
        self.sensor4_edt = QtWidgets.QLineEdit(self.environment_gb)
        self.sensor4_edt.setEnabled(False)
        self.sensor4_edt.setGeometry(QtCore.QRect(66, 132, 21, 21))
        self.sensor4_edt.setStyleSheet("background-color: rgb(130, 130, 130);")
        self.sensor4_edt.setObjectName("sensor4_edt")
        self.sensor_lbl_2 = QtWidgets.QLabel(self.environment_gb)
        self.sensor_lbl_2.setGeometry(QtCore.QRect(145, 31, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.sensor_lbl_2.setFont(font)
        self.sensor_lbl_2.setStyleSheet("color: rgb(255, 255, 255);")
        self.sensor_lbl_2.setAlignment(QtCore.Qt.AlignCenter)
        self.sensor_lbl_2.setObjectName("sensor_lbl_2")
        self.act3_lbl = QtWidgets.QLabel(self.environment_gb)
        self.act3_lbl.setGeometry(QtCore.QRect(155, 104, 17, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.act3_lbl.setFont(font)
        self.act3_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.act3_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.act3_lbl.setObjectName("act3_lbl")
        self.act1_edt = QtWidgets.QLineEdit(self.environment_gb)
        self.act1_edt.setEnabled(False)
        self.act1_edt.setGeometry(QtCore.QRect(180, 53, 21, 21))
        self.act1_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
        self.act1_edt.setObjectName("act1_edt")
        self.act2_lbl = QtWidgets.QLabel(self.environment_gb)
        self.act2_lbl.setGeometry(QtCore.QRect(155, 78, 17, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.act2_lbl.setFont(font)
        self.act2_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.act2_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.act2_lbl.setObjectName("act2_lbl")
        self.act4_edt = QtWidgets.QLineEdit(self.environment_gb)
        self.act4_edt.setEnabled(False)
        self.act4_edt.setGeometry(QtCore.QRect(180, 133, 21, 21))
        self.act4_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
        self.act4_edt.setObjectName("act4_edt")
        self.act2_edt = QtWidgets.QLineEdit(self.environment_gb)
        self.act2_edt.setEnabled(False)
        self.act2_edt.setGeometry(QtCore.QRect(180, 80, 21, 21))
        self.act2_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
        self.act2_edt.setObjectName("act2_edt")
        self.act4_lbl = QtWidgets.QLabel(self.environment_gb)
        self.act4_lbl.setGeometry(QtCore.QRect(155, 131, 17, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.act4_lbl.setFont(font)
        self.act4_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.act4_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.act4_lbl.setObjectName("act4_lbl")
        self.act1_lbl = QtWidgets.QLabel(self.environment_gb)
        self.act1_lbl.setGeometry(QtCore.QRect(155, 51, 17, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.act1_lbl.setFont(font)
        self.act1_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.act1_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.act1_lbl.setObjectName("act1_lbl")
        self.act3_edt = QtWidgets.QLineEdit(self.environment_gb)
        self.act3_edt.setEnabled(False)
        self.act3_edt.setGeometry(QtCore.QRect(180, 106, 21, 22))
        self.act3_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
        self.act3_edt.setObjectName("act3_edt")
        self.key2_pb = QtWidgets.QPushButton(self.environment_gb)
        self.key2_pb.setGeometry(QtCore.QRect(258, 83, 51, 21))
        self.key2_pb.setStyleSheet("color: rgb(255, 255, 255);\n"
"border-radius:10px;\n"
"background-color: rgb(167, 167, 167);\n"
"")
        self.key2_pb.setObjectName("key2_pb")
        self.key3_pb = QtWidgets.QPushButton(self.environment_gb)
        self.key3_pb.setGeometry(QtCore.QRect(258, 108, 51, 21))
        self.key3_pb.setStyleSheet("color: rgb(255, 255, 255);\n"
"border-radius:10px;\n"
"background-color: rgb(167, 167, 167);\n"
"\n"
"")
        self.key3_pb.setObjectName("key3_pb")
        self.key4_pb = QtWidgets.QPushButton(self.environment_gb)
        self.key4_pb.setGeometry(QtCore.QRect(258, 132, 51, 21))
        self.key4_pb.setStyleSheet("color: rgb(255, 255, 255);\n"
"border-radius:10px;\n"
"background-color: rgb(167, 167, 167);\n"
"")
        self.key4_pb.setObjectName("key4_pb")
        self.input_key_lbl = QtWidgets.QLabel(self.environment_gb)
        self.input_key_lbl.setGeometry(QtCore.QRect(258, 34, 55, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.input_key_lbl.setFont(font)
        self.input_key_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.input_key_lbl.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.input_key_lbl.setObjectName("input_key_lbl")
        self.key1_pb = QtWidgets.QPushButton(self.environment_gb)
        self.key1_pb.setGeometry(QtCore.QRect(258, 59, 51, 21))
        self.key1_pb.setStyleSheet("color: rgb(255, 255, 255);\n"
"border-radius:10px;\n"
"background-color: rgb(167, 167, 167);\n"
"")
        self.key1_pb.setObjectName("key1_pb")
        self.sensor1_lbl = QtWidgets.QLabel(self.environment_gb)
        self.sensor1_lbl.setGeometry(QtCore.QRect(44, 52, 16, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.sensor1_lbl.setFont(font)
        self.sensor1_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.sensor1_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.sensor1_lbl.setObjectName("sensor1_lbl")
        self.sensor2_lbl = QtWidgets.QLabel(self.environment_gb)
        self.sensor2_lbl.setGeometry(QtCore.QRect(44, 79, 16, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.sensor2_lbl.setFont(font)
        self.sensor2_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.sensor2_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.sensor2_lbl.setObjectName("sensor2_lbl")
        self.sensor3_lbl = QtWidgets.QLabel(self.environment_gb)
        self.sensor3_lbl.setGeometry(QtCore.QRect(44, 105, 16, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.sensor3_lbl.setFont(font)
        self.sensor3_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.sensor3_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.sensor3_lbl.setObjectName("sensor3_lbl")
        self.sensor4_lbl = QtWidgets.QLabel(self.environment_gb)
        self.sensor4_lbl.setGeometry(QtCore.QRect(44, 132, 16, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.sensor4_lbl.setFont(font)
        self.sensor4_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.sensor4_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.sensor4_lbl.setObjectName("sensor4_lbl")
        self.warning_start_btn_lbl = QtWidgets.QLabel(self.control_gb)
        self.warning_start_btn_lbl.setGeometry(QtCore.QRect(110, 745, 211, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.warning_start_btn_lbl.setFont(font)
        self.warning_start_btn_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.warning_start_btn_lbl.setText("")
        self.warning_start_btn_lbl.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.warning_start_btn_lbl.setWordWrap(True)
        self.warning_start_btn_lbl.setObjectName("warning_start_btn_lbl")
        self.warning_start_stop_lbl = QtWidgets.QLabel(self.control_gb)
        self.warning_start_stop_lbl.setGeometry(QtCore.QRect(110, 738, 471, 31))
        self.warning_start_stop_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.warning_start_stop_lbl.setText("")
        self.warning_start_stop_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.warning_start_stop_lbl.setObjectName("warning_start_stop_lbl")
        self.trial_lbl = QtWidgets.QLabel(self.control_gb)
        self.trial_lbl.setGeometry(QtCore.QRect(188, 704, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.trial_lbl.setFont(font)
        self.trial_lbl.setStyleSheet("color: rgb(255, 255, 255);")
        self.trial_lbl.setText("")
        self.trial_lbl.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.trial_lbl.setWordWrap(True)
        self.trial_lbl.setObjectName("trial_lbl")
        self.label_14 = QtWidgets.QLabel(self.control_gb)
        self.label_14.setGeometry(QtCore.QRect(580, 756, 101, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_14.setFont(font)
        self.label_14.setStyleSheet("color: rgb(211, 211, 211);")
        self.label_14.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # ******************** Modifications - Signals/Slots **************************
        
        # Change highlights after clicking:
        self.cam_connect_pb.setStyleSheet("QPushButton {color: rgb(0, 0, 0); border-radius: 10px; background-color: rgb(195, 195, 195);}"
                             "QPushButton::pressed {background-color : rgb(170, 114, 97); border-radius: 10px;}")
        
        self.cam_showpreview_pb.setStyleSheet("QPushButton {color: rgb(0, 0, 0); border-radius: 10px; background-color: rgb(195, 195, 195);}"
                             "QPushButton::pressed {background-color : rgb(170, 114, 97); border-radius: 10px;}")
        
        self.cam_takesnapshot_pb.setStyleSheet("QPushButton {color: rgb(0, 0, 0); border-radius: 10px; background-color: rgb(195, 195, 195);}"
                             "QPushButton::pressed {background-color : rgb(170, 114, 97); border-radius: 10px;}")
        
        self.ard_connect_pb.setStyleSheet("QPushButton {color: rgb(0, 0, 0); border-radius: 10px; background-color: rgb(195, 195, 195);}"
                             "QPushButton::pressed {background-color : rgb(170, 114, 97); border-radius: 10px;}")
        
        self.load_territories_pb.setStyleSheet("QPushButton {color: rgb(0, 0, 0); border-radius: 10px; background-color: rgb(195, 195, 195);}"
                             "QPushButton::pressed {background-color : rgb(170, 114, 97); border-radius: 10px;}")
        
        self.choose_behav_pb.setStyleSheet("QPushButton {color: rgb(0, 0, 0); border-radius: 10px; background-color: rgb(195, 195, 195);}"
                             "QPushButton::pressed {background-color : rgb(170, 114, 97); border-radius: 10px;}")
        
        self.load_segmodel_back_pb.setStyleSheet("QPushButton {color: rgb(0, 0, 0); border-radius: 10px; background-color: rgb(195, 195, 195);}"
                             "QPushButton::pressed {background-color : rgb(170, 114, 97); border-radius: 10px;}")
        
        self.load_segmodel_back_pb.setStyleSheet("QPushButton {color: rgb(0, 0, 0); border-radius: 10px; background-color: rgb(195, 195, 195);}"
                             "QPushButton::pressed {background-color : rgb(170, 114, 97); border-radius: 10px;}")
        
        self.choose_segmodel_deep_pb.setStyleSheet("QPushButton {color: rgb(0, 0, 0); border-radius: 10px; background-color: rgb(195, 195, 195);}"
                             "QPushButton::pressed {background-color : rgb(170, 114, 97); border-radius: 10px;}")
        
        self.create_segmodel_back_pb.setStyleSheet("QPushButton {color: rgb(0, 0, 0); border-radius: 10px; background-color: rgb(195, 195, 195);}"
                             "QPushButton::pressed {background-color : rgb(170, 114, 97); border-radius: 10px;}")
        
        self.open_folder_pb.setStyleSheet("QPushButton {color: rgb(0, 0, 0); border-radius: 10px; background-color: rgb(122, 122, 122);}"
                             "QPushButton::pressed {background-color : rgb(170, 114, 97); border-radius: 10px;}")
        
        self.key1_pb.setStyleSheet("QPushButton {color: rgb(255, 255, 255); border-radius: 10px; background-color: rgb(167, 167, 167);}"
                             "QPushButton::pressed {background-color : rgb(170, 114, 97); border-radius: 10px;}")
        
        self.key2_pb.setStyleSheet("QPushButton {color: rgb(255, 255, 255); border-radius: 10px; background-color: rgb(167, 167, 167);}"
                             "QPushButton::pressed {background-color : rgb(170, 114, 97); border-radius: 10px;}")
        
        self.key3_pb.setStyleSheet("QPushButton {color: rgb(255, 255, 255); border-radius: 10px; background-color: rgb(167, 167, 167);}"
                             "QPushButton::pressed {background-color : rgb(170, 114, 97); border-radius: 10px;}")
        
        self.key4_pb.setStyleSheet("QPushButton {color: rgb(255, 255, 255); border-radius: 10px; background-color: rgb(167, 167, 167);}"
                             "QPushButton::pressed {background-color : rgb(170, 114, 97); border-radius: 10px;}")
        
        self.start_pb.setStyleSheet("QPushButton {color: rgb(255, 255, 255); border-radius: 10px; background-color: rgb(117, 131, 113);}"
                             "QPushButton::pressed {background-color : rgb(170, 114, 97); border-radius: 10px;}")
        
        self.stop_pb.setStyleSheet("QPushButton {color: rgb(255, 255, 255); border-radius: 10px; background-color: rgb(85, 0, 0);}"
                             "QPushButton::pressed {background-color : rgb(170, 114, 97); border-radius: 10px;}")
        
        MainWindow.setWindowIcon(QtGui.QIcon('RatLogo.ico'))
        
        self.segparam_gb.setVisible(False)
        
        
        # Button: Connect camera
        self.allGood_camera = False
        self.cam_connect_pb.clicked.connect( self.cam_connect_pb_CLICKED )

        # Button: Show preview
        self.cam_showpreview_pb.clicked.connect( self.cam_showpreview_pb_CLICKED )
        
        # Button: take a snapshot
        self.cam_takesnapshot_pb.clicked.connect( self.cam_takesnapshot_pb_CLICKED )
        
        # Frames' parameters
        self.min_meters_sb.valueChanged.connect( self.min_meters_sb_CLICKED )
        
        self.MINValueDepthRange = 1024 #DEFAULT
        
        # Combo box: choose COM port
        COM_ports_available = serial.tools.list_ports.comports()
        self.allGood_arduino_connected = False
        self.allGood_arduino_streaming = False 
        self.COM_port = ''
        self.serial_port = serial.Serial()
        self.openSerialPort_counter = 0 # count number of times the serial port was open
        
        for port, desc, hwid in sorted(COM_ports_available):
            if desc.find('Arduino') != -1:
                self.COM_port = port
                self.allGood_arduino_connected = True
                
            else:
                self.COM_port = ''
                self.allGood_arduino_connected = False       
        
        # Button: Connect arduino
        self.ard_connect_pb.clicked.connect( self.ard_connect_pb_CLICKED )
        
        # list with all arduino outputs
        self.list_outputs_arduino = []

        # Button: Load deep learning model for behavior classification
        self.choose_behav_pb.clicked.connect( self.choose_behav_pb_CLICKED )
        
        self.allGood_modelBehav = True
        self.doBehaviorClassification = False
        
        # array to save depth frames [total number of frames, channels, img_size, img_size]
        self.N_STEPS = len(STEPS_LIST)
        self.array_depth = np.zeros((self.N_STEPS, CHANNELS_SIZE, IMG_SIZE, IMG_SIZE), dtype = np.float32)
        
        # array to save timesteps
        self.array_timesteps = []

        # List with all labels
        self.listLabels = []
        
        # List with all centroid outputs 
        self.list_centroids = []
        self.list_ROIs = []
        
        # List with key pressed
        self.keyPressed = []
        self.list_key_pressed = []

        #################### DEEP MODEL INITIALIZATION - BEHAV #######################
        
        self.device_behav = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Classification model
        model_behav = getattr(classificationModel_functions, f'Model_X_A')(self.N_STEPS)
        
        self.model_behav = model_behav
        self.modelname_behav_trained = ''
          
        #################### DEEP MODEL INITIALIZATION - SEGMENTATION #######################
                
        self.allGood_modelSeg = True
        self.doTracking = False
        self.allGood_goodSegMask = 1
        
        self.modelname_seg_trained = ''

        # Scale factor - coordinates centroid (from 128x128 -> 480x640)
        self.scl_x = ORIGINAL_SIZE_X / IMG_SIZE
        self.scl_y = ORIGINAL_SIZE_Y / IMG_SIZE

        self.choose_segmodel_deep_pb.setVisible(True)
        self.load_segmodel_back_pb.setVisible(False)
        self.create_segmodel_back_pb.setVisible(False)
        
        # warnings
        self.warning_segmodels_lbl.setVisible(True)
        self.warning_load_segmodelback.setVisible(False)
        self.warning_create_segmodel_back.setVisible(False)
        
        
        self.deep_model_rb.clicked.connect( self.deep_model_rb_CLICKED )
        self.backsub_model_rb.clicked.connect( self.backsub_model_rb_CLICKED )
        
        self.choose_segmodel_deep_pb.clicked.connect(self.choose_segmodel_deep_pb_CLICKED)

        #################### BACKGROUND SUBTRACTION - SEGMENTATION #######################
        # Load saved model
        self.load_segmodel_back_pb.clicked.connect(self.load_segmodel_back_pb_CLICKED)
        
        self.backmodelname_seg = ''
        self.modelBack = np.zeros((ORIGINAL_SIZE_Y, ORIGINAL_SIZE_X))
        
        # Create new model
        self.create_segmodel_back_pb.clicked.connect(self.create_segmodel_back_pb_CLICKED)
        
        self.allGood_createModel = False
        
        # Default min and max rat range - spin box
        self.minRatRange = MINRATRANGE
        self.maxRatRange = MAXRATRANGE
        
        ################### DIRECTORY TO SAVE ALL DATA ###################################
        self.open_folder_pb.clicked.connect(self.open_folder_pb_CLICKED)
        
        self.name_dir_to_save_data  = ''
        self.name_dir_to_save_data_frames = ''
        self.allGood_SaveFiles = False
        
        today = date.today()
        self.name_id_experiment = today.strftime("%d_%m_%Y")
        
        self.newTrial_nameFile = ''
        
        ################### mROIS ##################################################
        self.load_territories_pb.clicked.connect(self.load_territories_pb_CLICKED)
        self.mROI_image = np.zeros((ORIGINAL_SIZE_Y, ORIGINAL_SIZE_X))
        self.mROI_image_gray = np.zeros((ORIGINAL_SIZE_Y, ORIGINAL_SIZE_X))
        
        self.N_ROIs = 1
        
        self.allGood_ROI = False
                
        ################## START BUTTON ###########################################
        
        self.trialNumber = 1
        
        self.counter_Totalframes = 0
        
        # Timer to produce the clock
        self.now_time = 0
        self.timer_lcd = QTimer()
        self.timer_lcd.timeout.connect(self.showTime_LCD)
        
        # Timer to start tracking + classification
        self.timer_start_class_track = QTimer()
        self.timer_start_class_track.timeout.connect(self.timer_start_class_track_function)
        
        self.started_timer_track = False
        
        # Button start - click
        self.start_pb.clicked.connect(self.start_pb_CLICKED)
        
        # Construct Thread event for key press
        self.key_timer = QTimer()
        self.key_timer.timeout.connect(self.key_timer_function)
        
        # Pipeline for camera
        self.pipelineCameraRealTime = rs.pipeline()
        self.configCameraRealTime = rs.config()
        self.configCameraRealTime.enable_stream(rs.stream.depth, ORIGINAL_SIZE_X, ORIGINAL_SIZE_Y, rs.format.z16, 30)
        
        ################## STOP BUTTON ###########################################
        self.stop_pb.clicked.connect(self.stop_pb_CLICKED)
        
        ################## KEY BUTTONS ###########################################
        self.key1_pb.clicked.connect(self.key1_pb_CLICKED)
        self.key2_pb.clicked.connect(self.key2_pb_CLICKED)
        self.key3_pb.clicked.connect(self.key3_pb_CLICKED)
        self.key4_pb.clicked.connect(self.key4_pb_CLICKED)
        
        
        # ********************* UI components - do not touch! **************************
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    # ******************** Modifications - Signals/Slots **************************
    def cam_connect_pb_CLICKED(self):
        
        self.warning_cam_connect_lbl.setText("")
        
        ctx = rs.context()
        
        if len(ctx.devices) > 0:
            for d in ctx.devices:
                self.warning_cam_connect_lbl.setText("Connected to " + d.get_info(rs.camera_info.name))
                self.allGood_camera = True
                self.cam_showpreview_pb.setEnabled(True)
                self.cam_takesnapshot_pb.setEnabled(True)

        else:
            self.warning_cam_connect_lbl.setText("No Intel Device connected!")
            self.allGood_camera = False
            self.cam_showpreview_pb.setEnabled(False)
            self.cam_takesnapshot_pb.setEnabled(False)


    def cam_showpreview_pb_CLICKED(self):
        # Configure depth and color streams
        
        self.warning_cam_connect_lbl.setText('')
        
        if self.allGood_camera:
            
            if self.backsub_model_rb.isChecked():
                
                pipeline = rs.pipeline()
                config = rs.config()    
                config.enable_stream(rs.stream.depth, ORIGINAL_SIZE_X, ORIGINAL_SIZE_Y, rs.format.z16, 30)
                
                
                # Apply range and show to user
                minValueDepthRange = int(self.min_meters_sb.text())
                              
                # Get X and Y border
                self.X_border_pixels = int(self.X_border_sb.text())
                self.Y_border_pixels = int(self.Y_border_sb.text())
                       
                # Start streaming
                pipeline.start(config)
        
                while True:
                    # Wait for a coherent pair of frames: depth and color
                    frames = pipeline.wait_for_frames()
                    
                    depth_frame = frames.get_depth_frame()
                    
                    if not depth_frame:
                        continue
            
                    # Convert images to numpy arrays
                    depth_image = np.asanyarray(depth_frame.get_data())
                            
                    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                    # depth_grayscalemap = ((5*depth_image)%255).astype(np.uint8)
                    
                    depth_image[depth_image < minValueDepthRange] = minValueDepthRange
                    depth_image[depth_image > minValueDepthRange + 255] = minValueDepthRange + 255
            
                    # Rescale values [0, 255]
                    depth_grayscalemap = (((depth_image - (minValueDepthRange)) / ((minValueDepthRange + 255) - (minValueDepthRange)))*(255)).astype(np.uint8)
                    depth_grayscalemap = ((2*depth_image)%255).astype(np.uint8)
            
                    depth_grayscalemap[:, 0:self.X_border_pixels] = 0.0
                    depth_grayscalemap[:, ORIGINAL_SIZE_X - self.X_border_pixels:ORIGINAL_SIZE_X] = 0.0
                    
                    depth_grayscalemap[0:self.Y_border_pixels, :] = 0.0
                    depth_grayscalemap[ORIGINAL_SIZE_Y - self.Y_border_pixels:ORIGINAL_SIZE_Y, :] = 0.0
            
                    # Show images
                    cv2.namedWindow('Preview - depth frame: with pre-defined border and dynamic depth range', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('Preview - depth frame: with pre-defined border and dynamic depth range', depth_grayscalemap)
                    cv2.waitKey(1)
                
                    # Close window to abort streaming
                    if cv2.getWindowProperty('Preview - depth frame: with pre-defined border and dynamic depth range', cv2.WND_PROP_VISIBLE) <= 0 :
                        break
        
                # Stop streaming
                cv2.destroyAllWindows()
                pipeline.stop()
                
                
                
            else:
                pipeline = rs.pipeline()
                config = rs.config()    
                config.enable_stream(rs.stream.depth, ORIGINAL_SIZE_X, ORIGINAL_SIZE_Y, rs.format.z16, 30)
                
                
                # Apply range and show to user
                minValueDepthRange = int(self.min_meters_sb.text())
                       
                # Start streaming
                pipeline.start(config)
        
                while True:
                    # Wait for a coherent pair of frames: depth and color
                    frames = pipeline.wait_for_frames()
                    
                    depth_frame = frames.get_depth_frame()
                    
                    if not depth_frame:
                        continue
            
                    # Convert images to numpy arrays
                    depth_image = np.asanyarray(depth_frame.get_data())
                            
                    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                    # depth_grayscalemap = ((5*depth_image)%255).astype(np.uint8)
                    
                    depth_image[depth_image < minValueDepthRange] = minValueDepthRange
                    depth_image[depth_image > minValueDepthRange + 255] = minValueDepthRange + 255
            
                    # Rescale values [0, 255]
                    depth_grayscalemap = (((depth_image - (minValueDepthRange)) / ((minValueDepthRange + 255) - (minValueDepthRange)))*(255)).astype(np.uint8)
                    depth_grayscalemap = ((2*depth_image)%255).astype(np.uint8)
            
                    # Show images
                    cv2.namedWindow('Preview - depth frame: with pre-defined dynamic depth range', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('Preview - depth frame: with pre-defined dynamic depth range', depth_grayscalemap)
                    cv2.waitKey(1)
                
                    # Close window to abort streaming
                    if cv2.getWindowProperty('Preview - depth frame: with pre-defined dynamic depth range', cv2.WND_PROP_VISIBLE) <= 0 :
                        break
        
                # Stop streaming
                cv2.destroyAllWindows()
                pipeline.stop()

    def cam_takesnapshot_pb_CLICKED(self):
        # Configure depth and color streams
        
        self.warning_cam_connect_lbl.setText('')
        
        if self.allGood_camera:
            
            folder_selected_snapshot = str(QFileDialog.getExistingDirectory(None, "DeepCaT-Z: Select the folder to save snapshot", options=QFileDialog.DontUseNativeDialog ))
            
            QApplication.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
            QApplication.processEvents()
            
            pipeline2 = rs.pipeline()
            config2 = rs.config()    
            config2.enable_stream(rs.stream.depth, ORIGINAL_SIZE_X, ORIGINAL_SIZE_Y, rs.format.z16, 30)
            config2.enable_stream(rs.stream.color, ORIGINAL_SIZE_X, ORIGINAL_SIZE_Y, rs.format.bgr8, 30)
            
            # Start streaming
            pipeline2.start(config2)
            
            align = rs.align(rs.stream.depth)
    
            count_snap = 0
            
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames2 = pipeline2.wait_for_frames()
                
                aligned_frames = align.process(frames2)
                
                depth_frame2 = aligned_frames.get_depth_frame()
                color_frame2 = aligned_frames.get_color_frame()
                                
                if not depth_frame2 or not color_frame2:
                    continue
        
                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame2.get_data())
                color_image = np.asanyarray(color_frame2.get_data())
                        
                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                depth_grayscalemap = ((5*depth_image)%255).astype(np.uint8)
            
                count_snap = count_snap + 1
                # Close window to abort streaming
                if count_snap == 4 :
                    imageio.imwrite(folder_selected_snapshot + '/snapshot_DEPTH.png', depth_grayscalemap.astype(np.uint8))
                    imageio.imwrite(folder_selected_snapshot + '/snapshot_RGB.png', color_image.astype(np.uint8))
                    break
    
            # Stop streaming
            cv2.destroyAllWindows()
            pipeline2.stop()
            
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()
            
        else:
            self.warning_cam_connect_lbl.setText('No Intel Device connected!')
        
    def ard_connect_pb_CLICKED(self):
        
        self.warning_ard_connect_lbl.setText('')
                                             
        # 1st time opening the port
        if self.openSerialPort_counter != 0:
            self.serial_port.close()
                  
        if self.allGood_arduino_connected:
        
            self.serial_port = serial.Serial( baudrate = BAUDRATE, port = self.COM_port, timeout = TIMEOUT_ARDUINO )
            time.sleep(1) #give the connection a second to settle
            
            # GET ARDUINO ACKNOWLEDGMENT - check streaming between PC and Arduino board
            msg = self.serial_port.readline()
            if not msg :
                self.allGood_arduino_streaming = False

            else :
                self.allGood_arduino_streaming = True
    
            # Give feedback
            if self.allGood_arduino_connected & self.allGood_arduino_streaming:
                self.warning_ard_connect_lbl.setText('Arduino connected via port ' + self.COM_port)
            else:
                self.warning_ard_connect_lbl.setText('Problems connecting to Arduino!')
        
            self.openSerialPort_counter = self.openSerialPort_counter + 1
        
        else: 
            self.allGood_arduino_connected = False
            self.warning_ard_connect_lbl.setText('Problems connecting to Arduino!')
     
        
     
    def min_meters_sb_CLICKED(self):
        self.max_meters_sb.setValue(self.min_meters_sb.value() + 255)
        
        
    def choose_behav_pb_CLICKED (self):
         
         self.warning_deepbehav_lbl.setText('')

         model_filename = QFileDialog.getOpenFileName(None, 'DeepCaT-Z: Open the trained model file - CLASSIFICATION', options=QFileDialog.DontUseNativeDialog )
         
         if isinstance(model_filename, tuple):
             model_filename = model_filename[0]
         else:
            model_filename = str(model_filename) 

         self.modelname_behav_trained = model_filename
         
         try:
             self.model_behav.load_state_dict(torch.load(self.modelname_behav_trained, map_location=torch.device('cpu'))) #, strict=False)    
             self.model_behav = self.model_behav.to(self.device_behav)
             self.model_behav.eval()
                       
             self.allGood_modelBehav = True
             self.doBehaviorClassification = True
             self.warning_deepbehav_lbl.setText('Model sucessfully loaded.')
                    
         except:
             self.allGood_modelBehav = False
             self.doBehaviorClassification = False
             self.warning_deepbehav_lbl.setText('Bad model file! Try again')
             
    
    def deep_model_rb_CLICKED(self):
        
        self.choose_segmodel_deep_pb.setVisible(True)
        self.load_segmodel_back_pb.setVisible(False)
        self.create_segmodel_back_pb.setVisible(False)
        
        self.segparam_gb.setVisible(False)
        
        self.UNet_gb.setVisible(True)
        
        # warnings
        self.warning_segmodels_lbl.setVisible(True)
        self.warning_load_segmodelback.setVisible(False)
        self.warning_create_segmodel_back.setVisible(False)
        
    def backsub_model_rb_CLICKED(self):
        
        self.choose_segmodel_deep_pb.setVisible(False)
        self.load_segmodel_back_pb.setVisible(True)
        self.create_segmodel_back_pb.setVisible(True)
        
        self.segparam_gb.setVisible(True)
        
        self.UNet_gb.setVisible(False)
        
        # warnings
        self.warning_segmodels_lbl.setVisible(False)
        self.warning_load_segmodelback.setVisible(True)
        self.warning_create_segmodel_back.setVisible(True)
        
     
    def choose_segmodel_deep_pb_CLICKED(self):

        self.warning_segmodels_lbl.setText('')
        
        model_filename_segdeep = QFileDialog.getOpenFileName(None, 'DeepCaT-Z: Open the trained model file - SEGMENTATION', options=QFileDialog.DontUseNativeDialog)
        
        if isinstance(model_filename_segdeep, tuple):
            model_filename_segdeep = model_filename_segdeep[0]
        else:
            model_filename_segdeep = str(model_filename_segdeep) 
        
        self.modelname_seg_trained = model_filename_segdeep
        
        try:
            
            if self.UNet_rb.isChecked():
                self.model_seg = UNet_simple(IMG_SIZE)
                        
            elif self.ConvLSTM_rb.isChecked():
                self.model_seg = UNet_ConvLSTM('convlstm-v3', IMG_SIZE)
            
            self.model_seg = tf.keras.models.load_model(self.modelname_seg_trained, compile=False)
            self.allGood_modelSeg = True
            self.doTracking = True
            self.warning_segmodels_lbl.setText('Model sucessfully loaded.')
            
        except:
            self.allGood_modelSeg = False
            self.doTracking = False
            self.warning_segmodels_lbl.setText('Bad model file! Try again')
    
    def load_segmodel_back_pb_CLICKED(self):
         
        self.warning_load_segmodelback.setText('')
        self.warning_create_segmodel_back.setText('')
        
        backmodel_filename = QFileDialog.getOpenFileName(None, 'DeepCaT-Z: Load the background model file - SEGMENTATION', options=QFileDialog.DontUseNativeDialog)
        
        if isinstance(backmodel_filename, tuple):
            backmodel_filename = backmodel_filename[0]
        else:
            backmodel_filename = str(backmodel_filename)
            
        self.backmodelname_seg = backmodel_filename
        
        try:
            self.modelBack = imread(self.backmodelname_seg, as_gray = True)
            self.allGood_modelSeg = True
            self.doTracking = True
            self.warning_load_segmodelback.setVisible(True)
            self.warning_load_segmodelback.setText('Background model sucessfully loaded.')
            
        except:
            self.allGood_modelSeg = False
            self.doTracking = False
            self.warning_load_segmodelback.setVisible(True)
            self.warning_load_segmodelback.setText('Bad background model file! Try again')
        
    
    def msgbtn(self, i):
        
        button_clicked = i.text()
        
        if button_clicked.find('Yes') != -1:
            self.allGood_createModel = True
        else:
            self.allGood_createModel = False
            
    def create_segmodel_back_pb_CLICKED(self):
        
        if self.allGood_camera:
        
            self.warning_create_segmodel_back.setText('')
            self.warning_load_segmodelback.setText('')
            
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowIcon(QtGui.QIcon('RatLogo.ico'))
            
            msg.setText("Do you want to create and save a new background model?")
            msg.setWindowTitle("Background model - create")
            msg.setDetailedText("Once you click Yes, 200 frames will be acquired using the connected camera to create a new background model. Make sure the field of view only contains background pixels and keep the background clean, without any animal.")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.buttonClicked.connect(self.msgbtn)
            	
            msg.exec_()
            
            # Acquire frames to create model if everything is OK:
            if self.allGood_createModel:
    
                self.warning_create_segmodel_back.setText('')
                
                # Disable all other buttons
                self.cam_connect_pb.setEnabled(False)
                self.cam_showpreview_pb.setEnabled(False)
                self.ard_connect_pb.setEnabled(False)
                self.choose_behav_pb.setEnabled(False)
                self.deep_model_rb.setEnabled(False)
                self.backsub_model_rb.setEnabled(False)
                self.load_segmodel_back_pb.setEnabled(False)
                self.choose_segmodel_deep_pb.setEnabled(False)
                self.create_segmodel_back_pb.setEnabled(False)
                self.MinRange_sb.setEnabled(False)
                self.MaxRange_sb.setEnabled(False)
                self.open_folder_pb.setEnabled(False)
                self.load_territories_pb.setEnabled(False)
                self.start_pb.setEnabled(False)
                self.stop_pb.setEnabled(False)
                
                # Change cursor
                QApplication.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))


                pipeline = rs.pipeline()
                config = rs.config()    
                config.enable_stream(rs.stream.depth, ORIGINAL_SIZE_X, ORIGINAL_SIZE_Y, rs.format.z16, 30)
                
                # Start streaming
                pipeline.start(config)
        
                counter_frame = 0
                array_depthframes_toBack = np.zeros((ORIGINAL_SIZE_Y, ORIGINAL_SIZE_X, NUM_FRAMES_TO_CREATE_MODEL), dtype = np.float32)
                
                while counter_frame < NUM_FRAMES_TO_CREATE_MODEL:
                    # Wait for a coherent pair of frames: depth and color
                    frames = pipeline.wait_for_frames()
                    depth_frame = frames.get_depth_frame()
                    
                    if not depth_frame:
                        continue
            
                    # Convert images to numpy arrays
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    array_depthframes_toBack[:, :, counter_frame] = depth_image
                
                    counter_frame = counter_frame + 1

                # Stop streaming
                pipeline.stop()
                
                # Calculate median frame of array
                median_back_model = np.median(array_depthframes_toBack, axis = 2)
                
                # Save model in application for further use:
                self.modelBack = median_back_model                  
                
                # Visualize model
                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_grayscalemap = cv2.applyColorMap(cv2.convertScaleAbs(median_back_model, alpha=0.03), cv2.COLORMAP_JET)
                #depth_grayscalemap = ((5*median_back_model)%255).astype(np.uint8)
        
                # Return cursor to original shape
                QApplication.restoreOverrideCursor()
                
                # Show images
                cv2.namedWindow('Preview - background model frame', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Preview - background model frame', depth_grayscalemap)
                cv2.waitKey(0)
                          
                # Save model to folder
                imageio.imwrite('modelBack_new_'+ self.name_id_experiment + '.png', median_back_model.astype(np.uint16))
    
                self.allGood_modelSeg = True
                self.doTracking = True
            
                # Disable all other buttons
                self.cam_connect_pb.setEnabled(True)
                self.cam_showpreview_pb.setEnabled(True)
                self.ard_connect_pb.setEnabled(True)
                self.choose_behav_pb.setEnabled(True)
                self.deep_model_rb.setEnabled(True)
                self.backsub_model_rb.setEnabled(True)
                self.load_segmodel_back_pb.setEnabled(True)
                self.choose_segmodel_deep_pb.setEnabled(True)
                self.create_segmodel_back_pb.setEnabled(True)
                self.MinRange_sb.setEnabled(True)
                self.MaxRange_sb.setEnabled(True)
                self.open_folder_pb.setEnabled(True)
                self.load_territories_pb.setEnabled(True)
                self.start_pb.setEnabled(True)
                self.stop_pb.setEnabled(True)
                
                self.warning_create_segmodel_back.setVisible(True)
                self.warning_create_segmodel_back.setText('Model sucessfully created and saved.')
                
            else:
                self.warning_create_segmodel_back.setVisible(True)
                self.warning_create_segmodel_back.setText('No model created.')
                self.warning_load_segmodelback.setText('')
                self.allGood_createModel = False
            
        else:
            self.allGood_createModel = False
            self.warning_create_segmodel_back.setText('Connect a depth camera and try again!')
            self.warning_load_segmodelback.setText('')

    
    def open_folder_pb_CLICKED (self):

        self.directory_edt.setText('')
        self.warning_selectfolder_lbl.setText('')

        folder_selected = str(QFileDialog.getExistingDirectory(None, "DeepCaT-Z:Select the folder to save data", options=QFileDialog.DontUseNativeDialog ))
        
        self.name_dir_to_save_data  = folder_selected
        
        
        self.name_dir_to_save_data = self.name_dir_to_save_data + '/Experiment_' + self.name_id_experiment
        
        try:
            os.mkdir(self.name_dir_to_save_data)
            
            self.directory_edt.setText(self.name_dir_to_save_data)
            self.warning_selectfolder_lbl.setText('Ready to save files.')
            self.allGood_SaveFiles = True
            
            # Reset trial number
            self.trialNumber = 1
                
        except FileExistsError:
            self.warning_selectfolder_lbl.setText('Folder to save frames already exists. Choose another diretory!')
            self.allGood_SaveFiles = False

    
    def load_territories_pb_CLICKED(self):

        self.warning_load_territories_lbl.setText('# ROIs')
        
        mROI_filename = QFileDialog.getOpenFileName(None, 'DeepCaT-Z: Load the ROI file', options=QFileDialog.DontUseNativeDialog)
        
        if isinstance(mROI_filename, tuple):
            mROI_filename = mROI_filename[0]
        else:
            mROI_filename = str(mROI_filename)
        if len(mROI_filename) == 0: # empty string
            self.allGood_ROI = False
            self.warning_load_territories_lbl.setText('Bad ROI file!')
        
        else:    
            
            try:
                self.mROI_image = imread(mROI_filename)
                self.allGood_ROI = True
            except:
                self.allGood_ROI = False
            
            if self.allGood_ROI:
                # Count number of unique regions
                self.N_ROIs = len(np.unique(self.mROI_image))
                self.uniqueValues = np.unique(self.mROI_image)
                    
                
                # Update lbl with # of ROIs
                self.warning_load_territories_lbl.setText(str(self.N_ROIs) + ' ROIs')
                
                # Display ROI image on label
                shape_image = self.mROI_image.shape
                if len(shape_image) == 2:
                    channels_img = 1
                else:
                    channels_img = shape_image [2]
                
                height_img = shape_image [0]
                width_img = shape_image [1]
                bytesPerLine = channels_img * width_img
                qImg = QtGui.QImage(self.mROI_image.data, width_img, height_img, bytesPerLine, QtGui.QImage.Format_RGB888)
                pixmap01 = QPixmap.fromImage(qImg)
                pixmap_image = QPixmap(pixmap01)
                
                self.territories_preview_lbl.setPixmap(pixmap_image)
                self.territories_preview_lbl.setScaledContents(True)
                self.territories_preview_lbl.show()
    
                self.mROI_image_gray = cv2.cvtColor(self.mROI_image, cv2.COLOR_BGR2GRAY)    
            
            else:
                self.warning_load_territories_lbl.setText('Bad ROI file!')
        
            
    def start_pb_CLICKED(self):    
        
        if self.allGood_camera:
            
            if self.allGood_SaveFiles:
                
                if self.allGood_modelBehav:
                    
                    if self.allGood_modelSeg:
                        
                        self.warning_start_btn_lbl.setText('')
                        self.warning_start_stop_lbl.setText('')
                        
                        # Disable all other buttons
                        self.cam_connect_pb.setEnabled(False)
                        self.cam_showpreview_pb.setEnabled(False)
                        self.ard_connect_pb.setEnabled(False)
                        self.choose_behav_pb.setEnabled(False)
                        self.deep_model_rb.setEnabled(False)
                        self.backsub_model_rb.setEnabled(False)
                        self.load_segmodel_back_pb.setEnabled(False)
                        self.choose_segmodel_deep_pb.setEnabled(False)
                        self.create_segmodel_back_pb.setEnabled(False)
                        self.MinRange_sb.setEnabled(False)
                        self.MaxRange_sb.setEnabled(False)
                        self.open_folder_pb.setEnabled(False)
                        self.load_territories_pb.setEnabled(False)
                        
                        self.UNet_gb.setEnabled(False)
                                            
                        self.warning_start_stop_lbl.setText('')
                        
                        self.newTrial_nameFile = self.name_dir_to_save_data + '/Trial_' + format(self.trialNumber, '03d')
                                
                        try:
                            os.mkdir(self.newTrial_nameFile)
                            
                            self.name_dir_to_save_data_frames = self.newTrial_nameFile +'/original_frames_png_' + self.name_id_experiment
                                                    
                            try:
                                os.mkdir(self.name_dir_to_save_data_frames)
                                
                                # Everything is OK, lets start recording + analyzing + saving:
                                # Get range
                                self.MINValueDepthRange = int(self.min_meters_sb.text())
                                
                                # Real time acquisition
                                self.pipelineCameraRealTime.start(self.configCameraRealTime)
                                
                                self.started_timer_track = True
                                
                                # Start thread for key press event:
                                self.key_timer.start()
                                
                                # Start timers
                                self.now_time = 0
                                self.timer_lcd.start(1000)
                                self.timer_start_class_track.start()             
                                                                
                            except FileExistsError:
                                self.warning_start_btn_lbl.setText('Folder to save frames already exists. Choose another diretory!')
                                self.allGood_SaveFiles = False
                                
                        except FileExistsError:
                            self.warning_start_btn_lbl.setText('Folder to save frames already exists. Choose another diretory!')
                            self.allGood_SaveFiles = False
   
                    else:
                        self.warning_start_stop_lbl.setVisible(False)
                        self.warning_start_btn_lbl.setVisible(True)
                        self.warning_start_btn_lbl.setText('Select a valid segmentation model file or create a new one!')
                    
                else: 
                    self.warning_start_stop_lbl.setVisible(False)
                    self.warning_start_btn_lbl.setVisible(True)
                    self.warning_start_btn_lbl.setText('Select a valid classification model file!')
            else:
                self.warning_start_stop_lbl.setVisible(False)
                self.warning_start_btn_lbl.setVisible(True)
                self.warning_start_btn_lbl.setText('Select a valid directory to save data!')
                    
        else:
            self.warning_start_stop_lbl.setVisible(False)
            self.warning_start_btn_lbl.setVisible(True)
            self.warning_start_btn_lbl.setText('First, connect a camera!')
    
    
    def showTime_LCD(self): # QTimer
        
        runtime = "%02d:%02d:%02d" % (self.now_time/3600, self.now_time/60, self.now_time % 60)
        
        self.directory_edt_2.setText(runtime)
        self.now_time +=1
        
    def timer_start_class_track_function(self):

        # ############## START ######################
        
        if self.started_timer_track:

            # Use camera to acquire frames
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipelineCameraRealTime.wait_for_frames()
            depth_frame = frames.get_depth_frame()
                              
            # Timestamps of depth frames
            ts_depth = depth_frame.get_timestamp()
        
            # Convert images to numpy arrays
            depth_image_RAW = np.asanyarray(depth_frame.get_data())  
                    
            # PROCESS FRAME: new dynamic range + rescale values [0-255] + resize to [IMG_SIZE x IMG_SIZE] + save to array
            # New dynamic range
            depth_image = depth_image_RAW.copy()
            
            depth_image[depth_image < self.MINValueDepthRange] = self.MINValueDepthRange
            depth_image[depth_image > self.MINValueDepthRange + 255] = self.MINValueDepthRange + 255
        
            # Rescale values [0, 255]
            depth_image = (((depth_image - (self.MINValueDepthRange)) / ((self.MINValueDepthRange + 255) - (self.MINValueDepthRange)))*(255)).astype(np.uint8)
            
            # TO SHOW
            depth_image_toShow = depth_image.copy()
            
            ######################## SAVE .png FILES TO FOLDER ###########################################
            cv2.imwrite(self.name_dir_to_save_data_frames + '/' + str(self.counter_Totalframes) + '_uint8.png', depth_image_toShow)
            
            # Resize [256, 256]
            depth_image = ((resize(depth_image, (IMG_SIZE, IMG_SIZE), order = 0, preserve_range = True))/255).astype(np.float32)
            
            # Save frames to array:
            p_loc = self.counter_Totalframes % self.N_STEPS # location in array_depth
            
            self.array_depth[p_loc, 0, :, :] = depth_image
                    
            
            # First N frames just save
            if self.counter_Totalframes < self.N_STEPS -1:
                    
                self.counter_Totalframes = self.counter_Totalframes+1
                
            else:
                
                # Save timestamps
                self.array_timesteps.append(ts_depth)
                
                # Construct permute_list: list with permutation indexes of array_depth (to avoid reshaping array with the correct order)
                permute_list = [(p_loc+i)%self.N_STEPS for i in range (1, self.N_STEPS+1)]
                              
                list_new_order = np.array([self.array_depth[permute_list[k]] for k in range(self.N_STEPS)])
                
                
                # ************* DEEP LEARNING MODELS: BEHAVIOR CLASSIFICATION *****
                if self.doBehaviorClassification:
                    
                    # Predict behavioral class from images
                    # create testing data list - with the correct form to enter the model
                    testing_data_list = [(torch.from_numpy(np.ascontiguousarray(list_new_order)).unsqueeze(0))]
                    
                    # GPU machine:
                    if GPU_AVAILABLE:
                        testing_data_list = torch.cat(testing_data_list, dim=0)
                        testing_data_list = testing_data_list.to(self.device_behav) 
                        
                    # Predict
                    with torch.no_grad():
                        # Ask model for prediction
                        test_preds = self.model_behav(testing_data_list[0])
                        
                        # Get class with highest probability
                        test_preds_label = ((test_preds[0].argmax(1)).cpu()).numpy()
                        
                    # print(test_preds_label[0])
                    self.listLabels.append(test_preds_label[0])
                
                else: # no classification -> signal: NaN
                    test_preds_label = [-1]
                    self.listLabels.append(test_preds_label[0])

                # ************* AUTOMATIC SEGMENTATION ***************************
                if self.doTracking:
                    
                    if self.deep_model_rb.isChecked():
                        # OPTION 1 - DEEP LEARNING SEMANTIC SEGMENTATION METHOD 
                        
                        if self.UNet_rb.isChecked(): # - UNet with NO TIME DISTRIBUTED LAYERS
                            # Reshape to [batch, steps, size, size, channels]
                            testing_data_list_segs = (list_new_order.transpose(0, 2, 3, 1))[np.newaxis, ...]
                        
                            testing_data_list_segs = testing_data_list_segs[0, -1, :, :, :][np.newaxis, ...]
                        
                        elif self.ConvLSTM_rb.isChecked(): # - UNet with ConvLSTM v3
                            
                            # Reshape to [batch, steps, size, size, channels]
                            testing_data_list_segs = (list_new_order.transpose(0, 2, 3, 1))[np.newaxis, ...]
                        
                        # Predict
                        Yhat_test = self.model_seg.predict(testing_data_list_segs)
                        
                        # Create bin image - mask
                        Yhat_test_bin = np.array(Yhat_test>0.5, dtype=float)
                        
                        # Pre-process
                        img = Yhat_test_bin[0, :, :, :].squeeze().astype('uint8')
                             
                        # Connected components
                        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
                        sizes = stats[1:, -1] # eliminate background (1st object always)
                        nb_components = nb_components - 1
                    
                        if nb_components == 0: # no components = no animal -> bad segmentation
                            goodSegmentation = 0
                        
                        else: 
                            img2 = np.zeros((output.shape))
                              
                            # save the biggest object
                            id_maxObject= sizes.argmax()
                            img2[output == id_maxObject + 1] = 1
                            
                            # Erosion: to remove the tail or some adjacent pixels -> more stable centroid
                            img2_erode = cv2.erode(img2, KERNEL_EROSION, iterations=1)
                            
                            nb_components, output, stats_erode, centroids2 = cv2.connectedComponentsWithStats(img2_erode.astype('uint8'), connectivity=8)
                            sizes_erode = stats_erode[1:, -1]
                            id_maxObject_erode = sizes_erode.argmax()
                            
                            if nb_components == 0: # no components = no animal -> bad segmentation
                                centroidsAnimal = np.round((centroids[id_maxObject + 1])).astype('uint8') # too much erosion
                                Y_bin_final_filtered = img2.reshape(IMG_SIZE, IMG_SIZE, 1).astype('int32')
                                goodSegmentation = 1
                                
                            else:
                                centroidsAnimal = np.round((centroids2[id_maxObject_erode + 1])).astype('uint8') # +1 to eliminate background again
                                Y_bin_final_filtered = img2.reshape(IMG_SIZE, IMG_SIZE, 1).astype('int32')
                                goodSegmentation = 1
                        
                        if goodSegmentation:
                            # Get position x, y, z & zone
                            pos_x = (round(centroidsAnimal[0]*self.scl_x))
                            pos_y = (round(centroidsAnimal[1]*self.scl_y))
                            pos_z = (depth_image_RAW[pos_y, pos_x]) #get z value from the original depth frame (480x640 last frame to be acquired)
                            self.list_centroids.append([pos_x, pos_y, pos_z])
                        else:
                            pos_x = -1
                            pos_y = -1
                            pos_z = -1
                            self.list_centroids.append([pos_x, pos_y, pos_z])
                        
                    elif self.backsub_model_rb.isChecked():
                        # OPTION 2 - BACKGROUND SUBTRACTION METHOD - with BACKGROUND model                                
                        diff_image = np.absolute(np.subtract(self.modelBack.astype(np.float32), depth_image_RAW.astype(np.float32)))
                        
                        # Get minRatRange/maxRatRange from UI
                        self.minRatRange = int(self.MinRange_sb.text())
                        self.maxRatRange = int(self.MaxRange_sb.text())
                        
                        if self.minRatRange > self.maxRatRange: # ERROR. set default values
                            self.minRatRange = 5
                            self.maxRatRange = 300
                        
                        elif self.minRatRange < 0 | self.maxRatRange > 1000:
                            self.minRatRange = 5
                            self.maxRatRange = 300
                          
                        diff_image[diff_image < self.minRatRange] = 0.0
                        diff_image[diff_image > self.maxRatRange] = 0.0
                        
                        # Get X and Y border
                        self.X_border_pixels = int(self.X_border_sb.text())
                        self.Y_border_pixels = int(self.Y_border_sb.text())
                            
                        diff_image[:, 0:self.X_border_pixels] = 0.0
                        diff_image[:, ORIGINAL_SIZE_X - self.X_border_pixels:ORIGINAL_SIZE_X] = 0.0
                        
                        diff_image[0:self.Y_border_pixels, :] = 0.0
                        diff_image[ORIGINAL_SIZE_Y - self.Y_border_pixels:ORIGINAL_SIZE_Y, :] = 0.0
                        
                        # Pre-process            
                        img_mask = diff_image.astype('uint8')
                        
                        # Connected components
                        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_mask, connectivity=4)
                        sizes = stats[1:, -1] # eliminate background (1st object always)
                        nb_components = nb_components - 1
                    
                        if nb_components == 0: # no components = no animal -> bad segmentation
                            goodSegmentation = 0
                        
                        else: 
                            img2 = np.zeros((output.shape))
                
                            # save the biggest object
                            id_maxObject= sizes.argmax()
                            img2[output == id_maxObject + 1] = 1
                            
                            img2_erode = cv2.erode(img2, KERNEL_EROSION, iterations=2)  
                            
                            nb_components, output, stats_erode, centroids2 = cv2.connectedComponentsWithStats(img2_erode.astype('uint8'), connectivity=4)
                            sizes_erode = stats_erode[1:, -1]
                            id_maxObject_erode = sizes_erode.argmax()
                                                
                            if nb_components == 0: # no components = no animal -> bad segmentation
                                centroidsAnimal = np.round((centroids[id_maxObject + 1])) # too much erosion
                                
                                goodSegmentation = 1
                                
                            else:
                                centroidsAnimal = np.round((centroids2[id_maxObject_erode + 1])) # +1 to eliminate background again
                                
                                goodSegmentation = 1
                        
                        if goodSegmentation:
                            # Get position x, y, z & zone
                            pos_x = round(centroidsAnimal[0])
                            pos_y = round(centroidsAnimal[1])
                            pos_z = (depth_image_RAW[pos_y, pos_x]) #get z value from the original depth frame
                            self.list_centroids.append([pos_x, pos_y, pos_z])
                        else:
                            pos_x = -1
                            pos_y = -1
                            pos_z = -1
                            self.list_centroids.append([pos_x, pos_y, pos_z])
                
                else: # no not do tracking -> coords = [-1, -1, -1]
                    pos_x = -1
                    pos_y = -1
                    pos_z = -1
                    self.list_centroids.append([pos_x, pos_y, pos_z])
                    
                # Detect which zone the centroid is
                if self.doTracking:
                    if self.allGood_ROI:
                        value_ROI_image_centroid = self.mROI_image[pos_y, pos_x]
                        
                        zone_l = (np.argwhere(self.uniqueValues == value_ROI_image_centroid[0]))[0,0]
                                                                             
                    else:
                        zone_l = -1 #None
                else:
                    zone_l = -1
                
                self.list_ROIs.append(zone_l)
                                            
                # Get id if any key was pressed
                # keypress = -1;
                if self.keyPressed:
                    keypress_to_arduino = self.keyPressed[-1]
                    self.list_key_pressed.append(keypress_to_arduino)
                else:
                    keypress_to_arduino = -1
                    self.list_key_pressed.append(keypress_to_arduino)
                
                self.keyPressed.clear()
                
                # ################ SHOW IMAGE + TRACKING IN LABEL #######################################         
                depth_image_toShow_RGB = cv2.cvtColor(depth_image_toShow,cv2.COLOR_GRAY2RGB)
                
                # Draw a cross
                line_thickness = 1
                cv2.line(depth_image_toShow_RGB, (pos_x, 0), (pos_x, ORIGINAL_SIZE_Y), (0, 0, 0), thickness=line_thickness)
                cv2.line(depth_image_toShow_RGB, (0, pos_y), (ORIGINAL_SIZE_X, pos_y), (0, 0, 0), thickness=line_thickness)
                                            
                shape_image = depth_image_toShow_RGB.shape
                
                if len(shape_image) == 2:
                    channels_img = 1
                else:
                    channels_img = shape_image [2]

                height_img = shape_image [0]
                width_img = shape_image [1]
                bytesPerLine = channels_img * width_img
                
                qImg = QtGui.QImage(depth_image_toShow_RGB, width_img, height_img, bytesPerLine, QtGui.QImage.Format_RGB888)
                pixmap01 = QPixmap.fromImage(qImg)
                pixmap_image = QPixmap(pixmap01)
                
                self.preview_realtime_lbl.setPixmap(pixmap_image)
                self.preview_realtime_lbl.setScaledContents(True)
                self.preview_realtime_lbl.show()
                QApplication.processEvents()
                
                
                # # ************* ARDUINO - send and get message *******************               
                if self.allGood_arduino_connected & self.allGood_arduino_streaming:
                    
                    # Behavioral dictionary: 0 - standstill; 1 - walking; 2 - rearing; 3 - grooming
                    if test_preds_label[0] == 0:
                        state_arduino = 'S'
                    elif test_preds_label[0] == 1:
                        state_arduino = 'W'
                    elif test_preds_label[0] == 2:
                        state_arduino = 'R'
                    elif test_preds_label[0] == 3:
                        state_arduino = 'G'
                    elif test_preds_label[0] == -1:
                        state_arduino = 'N' #None
                    
                    # Create input message to arduino in the form: "state_arduino,pos_x,pos_y,pos_z,zone,keypress"
                    input_message = state_arduino + ',' + str(pos_x) + ',' + str(pos_y) + ',' + str(pos_z) + ',' + str(zone_l) + ',' + str(keypress_to_arduino) + '\n'
                    
                    # Send to arduino the input message in the form: "state_arduino,pos_x,pos_y,pos_z,zone,keypress"
                    self.serial_port.write(str.encode(input_message))
                                                        
                    # Get output - Input1, Input2, Output1, Output2
                    output_arduino = self.serial_port.readline()
                                                            
                    ################# UPDATE ENVIRONMENT VARIABLES ####################
                    # Get variables - Inputs and Outputs
                    if len(output_arduino) == 0:
                        
                        output_arduino = ''
                        self.list_outputs_arduino.append(output_arduino)
                    else:
                        
                        output_arduino_str = str(output_arduino)[2:-5]
                        self.list_outputs_arduino.append(output_arduino_str)
                        
                        # Input 1 - Sensor #1
                        input_sensor1 = output_arduino_str[0]
                        
                        if int(input_sensor1) == 0:
                            self.sensor1_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
                        else:
                            self.sensor1_edt.setStyleSheet("background-color: rgb(0, 255, 0);")
                        QApplication.processEvents()
                        
                        # Input 2 - Sensor #2
                        input_sensor2 = output_arduino_str[2]
                        
                        if int(input_sensor2) == 0:
                            self.sensor2_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
                        else:
                            self.sensor2_edt.setStyleSheet("background-color: rgb(0, 255, 0);")
                        QApplication.processEvents()
                        
                        # Input 3 - Sensor #3
                        input_sensor3 = output_arduino_str[4]
                        if int(input_sensor3) == 0:
                            self.sensor3_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
                        else:
                            self.sensor3_edt.setStyleSheet("background-color: rgb(0, 255, 0);")
                        QApplication.processEvents()
                        
                        # Input 4 - Sensor #4
                        input_sensor4 = output_arduino_str[6]
                        if int(input_sensor4) == 0:
                            self.sensor4_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
                        else:
                            self.sensor4_edt.setStyleSheet("background-color: rgb(0, 255, 0);")
                        QApplication.processEvents() 
                        
                        # Output 1 - Actuator #1 (LED)
                        output_actuator1 = output_arduino_str[8]
                        
                        if int(output_actuator1) == 0:
                            self.act1_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
                        else:
                            self.act1_edt.setStyleSheet("background-color: rgb(0, 255, 0);")
                        QApplication.processEvents()
                        
                        # Output 2 - Actuator #2
                        output_actuator2 = output_arduino_str[10]
                        
                        if int(output_actuator2) == 0:
                            self.act2_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
                        else:
                            self.act2_edt.setStyleSheet("background-color: rgb(0, 255, 0);")
                        QApplication.processEvents()
                        
                        # Output 3 - Actuator #3
                        output_actuator3 = output_arduino_str[12]
                        if int(output_actuator3) == 0:
                            self.act3_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
                        else:
                            self.act3_edt.setStyleSheet("background-color: rgb(0, 255, 0);")
                        QApplication.processEvents()
                        
                        # Output 4 - Actuator #4
                        output_actuator4 = output_arduino_str[14]
                        if int(output_actuator4) == 0:
                            self.act4_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
                        else:
                            self.act4_edt.setStyleSheet("background-color: rgb(0, 255, 0);")
                        QApplication.processEvents()
                    
                
                # ************* FINAL UPDATES ************************************
                self.counter_Totalframes += 1   
        
    def key_timer_function(self):
        if keyboard.is_pressed("1"):
            self.key1_pb.animateClick()
            QApplication.processEvents()
        
        if keyboard.is_pressed("2"):
            self.key2_pb.animateClick()
            QApplication.processEvents()
        
        if keyboard.is_pressed("3"):
            self.key3_pb.animateClick()
            QApplication.processEvents() 
        
        if keyboard.is_pressed("4"):
            self.key4_pb.animateClick()
            QApplication.processEvents()
        
    def stop_pb_CLICKED(self):
        
        # STOP TIMER - TRACKING + CLASSIFICATION
        self.started_timer_track = False
        self.timer_start_class_track.stop()
        QApplication.processEvents()
        
        self.pipelineCameraRealTime.stop()
        QApplication.processEvents()
        
        # Stop key press thread
        self.key_timer.stop()

        # STOP TIMER - CLOCK
        self.timer_lcd.stop()
        QApplication.processEvents()
        
        runtime = "%02d:%02d:%02d" % (0, 0, 0)
        self.directory_edt_2.setText(runtime)
        QApplication.processEvents()
        
        self.trialNumber = self.trialNumber + 1
          
        # Save data to files
        # Save timestamp + behavior + centroid + ROI + keypress + arduino output
        array_timesteps_tosave = np.array(self.array_timesteps).reshape((-1,1))
        for id in range(len(array_timesteps_tosave)):
            array_timesteps_tosave[id] = np.round(array_timesteps_tosave[id], decimals = 1)
         
        listLabels_tosave = np.array(self.listLabels).reshape((-1,1))
        
        list_centroids_tosave = np.array(self.list_centroids).reshape((-1,3))
        list_ROI_tosave = np.array(self.list_ROIs).reshape((-1,1))
        list_key_pressed_tosave = np.array(self.list_key_pressed).reshape((-1,1))

        if self.allGood_arduino_connected & self.allGood_arduino_streaming:
            
            list_append_arduino = []
            
            for ind in self.list_outputs_arduino:
                
                split_string = ind.split(',')
                list_append_arduino = list_append_arduino + split_string        
            
            list_outputs_arduino_tosave = np.array(list_append_arduino).reshape((-1,8))
 
            appendList = np.hstack((array_timesteps_tosave, listLabels_tosave, list_centroids_tosave, list_ROI_tosave, list_key_pressed_tosave, list_outputs_arduino_tosave))
            allData_tosave_df = pd.DataFrame (appendList, columns = ['timesteps [ms]', 'behavior label', 'centroid x [px]', 'centroid y [px]', 'centroid z [mm]', 'ROI id', 'key pressed', 'output arduino: DI_0', 'DI_1', 'DI_2', 'DI_3', 'DO_0', 'DO_1', 'DO_2', 'D0_3'])
         
        else:
            appendList = np.hstack((array_timesteps_tosave, listLabels_tosave, list_centroids_tosave, list_ROI_tosave, list_key_pressed_tosave))
            allData_tosave_df = pd.DataFrame (appendList, columns = ['timesteps [ms]', 'behavior label', 'centroid x [px]', 'centroid y [px]', 'centroid z [mm]', 'ROI id', 'key pressed'])
        
        allData_tosave_df.to_csv(self.newTrial_nameFile + '/experimental_data.csv')
        
        
        # Reset vectors/Arrays/Lists
        self.array_timesteps.clear()
        self.listLabels.clear()
        self.list_centroids.clear()
        self.list_ROIs.clear()
        self.list_key_pressed.clear()
        self.list_outputs_arduino.clear()
        self.counter_Totalframes = 0
        
        # Disable all other buttons
        self.cam_connect_pb.setEnabled(True)
        self.cam_showpreview_pb.setEnabled(True)
        self.ard_connect_pb.setEnabled(True)
        self.choose_behav_pb.setEnabled(True)
        self.deep_model_rb.setEnabled(True)
        self.backsub_model_rb.setEnabled(True)
        self.load_segmodel_back_pb.setEnabled(True)
        self.choose_segmodel_deep_pb.setEnabled(True)
        self.create_segmodel_back_pb.setEnabled(True)
        self.MinRange_sb.setEnabled(True)
        self.MaxRange_sb.setEnabled(True)
        self.open_folder_pb.setEnabled(True)
        self.load_territories_pb.setEnabled(True)
        self.UNet_gb.setEnabled(True)
        
        self.preview_realtime_lbl.clear()
        self.warning_start_btn_lbl.setVisible(False)
        self.warning_start_stop_lbl.setText('Acquisition and analysis stopped. Experimental data saved!')
        
        self.warning_load_segmodelback.setVisible(False)
        self.warning_create_segmodel_back.setVisible(False)
      
        self.act1_edt.setStyleSheet("background-color: rgb(122, 122, 122);") 
        self.act2_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
        self.act3_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
        self.act4_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
        self.sensor1_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
        self.sensor2_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
        self.sensor3_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
        self.sensor4_edt.setStyleSheet("background-color: rgb(122, 122, 122);")
        
    def key1_pb_CLICKED(self):
        self.keyPressed.append(1)
        QApplication.processEvents()
    
    def key2_pb_CLICKED(self):
        self.keyPressed.append(2)
        QApplication.processEvents()
    
    def key3_pb_CLICKED(self):
        self.keyPressed.append(3)
        QApplication.processEvents()
        
    def key4_pb_CLICKED(self):
        self.keyPressed.append(4)
        QApplication.processEvents()
     
        
    # ********************* UI components - do not touch! **************************

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DeepCaT-z: Software for real-time behavior recognition and automated closed-loop control of behavioral experiments"))
        self.setup_lbl.setText(_translate("MainWindow", "SETUP"))
        self.cam_connection_lbl.setText(_translate("MainWindow", "CAMERA"))
        self.cam_connect_pb.setText(_translate("MainWindow", "CONNECT"))
        self.cam_showpreview_pb.setText(_translate("MainWindow", "SHOW PREVIEW"))
        self.cam_takesnapshot_pb.setText(_translate("MainWindow", "TAKE SNAPSHOT"))
        self.ard_connection_lbl.setText(_translate("MainWindow", "ARDUINO"))
        self.ard_connect_pb.setText(_translate("MainWindow", "CONNECT"))
        self.ard_connection_lbl_2.setText(_translate("MainWindow", "ARDUINO CONNECTION"))
        self.deep_lbl.setText(_translate("MainWindow", "DEEP LEARNING MODELS"))
        self.choose_behav_pb.setText(_translate("MainWindow", "LOAD"))
        self.classific_lbl.setText(_translate("MainWindow", "Classification model:"))
        self.segment_lbl.setText(_translate("MainWindow", "Segmentation model:"))
        self.segment_lbl_2.setText(_translate("MainWindow", "Segmentation parameters - optional"))
        self.X_lbl.setText(_translate("MainWindow", "X [px]:"))
        self.border_lbl.setText(_translate("MainWindow", "ROI Border)"))
        self.Y_lbl.setText(_translate("MainWindow", "Y [px]:"))
        self.min_lbl.setText(_translate("MainWindow", "min [mm]:"))
        self.max_lbl.setText(_translate("MainWindow", "max [mm]:"))
        self.border_lbl_2.setText(_translate("MainWindow", "Range after subtraction)"))
        self.choose_segmodel_deep_pb.setText(_translate("MainWindow", "LOAD"))
        self.load_segmodel_back_pb.setText(_translate("MainWindow", "LOAD"))
        self.create_segmodel_back_pb.setText(_translate("MainWindow", "CREATE NEW"))
        self.backsub_model_rb.setText(_translate("MainWindow", "with background subtraction"))
        self.deep_model_rb.setText(_translate("MainWindow", "with deep learning"))
        self.ConvLSTM_rb.setText(_translate("MainWindow", "UNet + ConvLSTM"))
        self.UNet_rb.setText(_translate("MainWindow", "UNet"))
        self.save_param_lbl.setText(_translate("MainWindow", "SAVE DATA"))
        self.open_folder_pb.setText(_translate("MainWindow", "..."))
        self.selectfolder_lbl.setText(_translate("MainWindow", "Select directory:"))
        self.ard_connection_lbl_3.setText(_translate("MainWindow", "FRAMES\' PARAMETERS"))
        self.ard_connection_lbl_4.setText(_translate("MainWindow", "ARDUINO CONNECTION"))
        self.deptherror_lbl.setText(_translate("MainWindow", "Depth working range)"))
        self.min_meters_lbl.setText(_translate("MainWindow", "min [mm]:"))
        self.max_meters_lbl.setText(_translate("MainWindow", "max [mm]:"))
        self.setup_lbl_3.setText(_translate("MainWindow", "BEHAVIORAL EXPERIMENTS"))
        self.territories_lbl.setText(_translate("MainWindow", "REGIONS OF INTEREST"))
        self.load_territories_pb.setText(_translate("MainWindow", "LOAD IMAGE"))
        self.warning_load_territories_lbl.setText(_translate("MainWindow", "# ROIs"))
        self.start_pb.setText(_translate("MainWindow", "START"))
        self.stop_pb.setText(_translate("MainWindow", "STOP"))
        self.label_3.setText(_translate("MainWindow", "TIMER:"))
        self.directory_edt_2.setText(_translate("MainWindow", "00:00:00"))
        self.environment_lbl.setText(_translate("MainWindow", "ENVIRONMENT VARIABLES"))
        self.sensor_lbl.setText(_translate("MainWindow", "Sensors"))
        self.sensor_lbl_2.setText(_translate("MainWindow", "Actuators"))
        self.act3_lbl.setText(_translate("MainWindow", "# 3"))
        self.act2_lbl.setText(_translate("MainWindow", "# 2"))
        self.act4_lbl.setText(_translate("MainWindow", "# 4"))
        self.act1_lbl.setText(_translate("MainWindow", "# 1"))
        self.key2_pb.setText(_translate("MainWindow", "key 2"))
        self.key3_pb.setText(_translate("MainWindow", "key 3"))
        self.key4_pb.setText(_translate("MainWindow", "key 4"))
        self.input_key_lbl.setText(_translate("MainWindow", "Input keys"))
        self.key1_pb.setText(_translate("MainWindow", "key 1"))
        self.sensor1_lbl.setText(_translate("MainWindow", "# 1"))
        self.sensor2_lbl.setText(_translate("MainWindow", "# 2"))
        self.sensor3_lbl.setText(_translate("MainWindow", "# 3"))
        self.sensor4_lbl.setText(_translate("MainWindow", "# 4"))
        self.label_14.setText(_translate("MainWindow", "AG & PA. 2021"))

# Initialization - APP 
if __name__ == "__main__" :
    
    app = QtWidgets.QApplication( sys.argv )
    MyMain = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MyMain)
    MyMain.show()
    ret = app.exec_()
  
    sys.exit( ret )