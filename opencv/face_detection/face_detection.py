import numpy as np
import cv2
import pandas as pd
import os
import pickle

#model from cafee
face_detection_model = './models/res10_300x300_ssd_iter_140000.caffemodel'
face_detection_proto = './models/deploy.prototxt.txt'
#model from torch
face_descriptor = './models/openface.nn4.small2.v1.t7'
# load models using cv2 dnn
detector_model = cv2.dnn.readNetFromCaffe(face_detection_proto,face_detection_model)
descriptor_model = cv2.dnn.readNetFromTorch(face_descriptor)


# consider sample image
img = cv2.imread('./images/Sachin Tendulkar/2200.jpg')
cv2.imshow('sample',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# face detection on copy of image 

image = img.copy()
h,w = image.shape[:2]

print(h)
print(w)
img_blob = cv2.dnn.blobFromImage(image,1,(300,300),(104,177,123),swapRB=False,crop=False)

detector_model.setInput(img_blob)
detections = detector_model.forward()
# detections is the faces, len(detections) give the number pf faces

print(len(detections))

if len(detections) > 0 :
     # detections[0,0,:,2]  is the table with all the probability
     # knowing that the model detect many faces in one blob, we have to get the face with the max probability to maximize our chace to detect a face
     i = np.argmax(detections[0,0,:,2])
     confidence = detections[0,0,i,2]
     if confidence > 0.5 :
            print(detections[0,0,i,2])
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startx,starty,endx,endy) = box.astype('int')
            # step-2: Feature Extraction or Embedding
            img_draw = image.copy()
            cv2.rectangle(img_draw,(startx,starty),(endx,endy),(255,0,0))
            
cv2.imshow('image draw',img_draw)
cv2.waitKey(0)
