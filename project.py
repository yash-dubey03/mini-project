import cv2
import numpy as np
import random
import os
from PIL import Image
import time
import imutils
from tensorflow.keras.models import load_model

#camera usage

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


model = load_model('helmet-nonhelmet_cnn.h5')
print('model loaded!!!')

cap = cv2.VideoCapture('testing videos/test2.mp4')
COLORS = [(0,255,0),(0,0,255)]

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter('output.avi', fourcc, 5,(888,500))
writer = VideoWriter('output.avi',(frame.shape[1], frame.shape[0]))
writer.open()
