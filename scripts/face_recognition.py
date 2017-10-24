#!/usr/bin/env python
#-*- coding: UTF-8 -*- 

from __future__ import print_function
import rospy
import cv2
from std_msgs.msg import String
import os,dlib,glob,numpy
from skimage import io
import os
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

rospy.init_node('face_recognition', anonymous=True)

def face_recognition():
    predictor_path = "/home/pedestrian-username/catkin_ws/src/face_recognition/scripts/shape_predictor_68_face_landmarks.dat"
    face_rec_model_path = "/home/pedestrian-username/catkin_ws/src/face_recognition/scripts/dlib_face_recognition_resnet_model_v1.dat"
    faces_folder_path = "/home/pedestrian-username/catkin_ws/src/face_recognition/scripts/candidate"
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    descriptors = []
    candidate = []
    for f in glob.glob(os.path.join(faces_folder_path,"*.jpg")):
        candidate.append(f.split("candidate/")[1].split(".")[0])
        print("Processing file: {}".format(f))
        img = io.imread(f)
        dets = detector(img,1)
        print("Number of faces detected:{}".format(len(dets)))
        for k,d in enumerate(dets):
            shape = sp(img,d)
            face_descriptor = facerec.compute_face_descriptor(img,shape)
            v = numpy.array(face_descriptor)
            descriptors.append(v)
    cv2.namedWindow('采集图像')
    cap=cv2.VideoCapture(0)
    cap.set(3,1000)
    cap.set(4,800)
    while not rospy.is_shutdown():
        ret,frame=cap.read()
        frame=cv2.flip(frame, 1)
        dets = detector(frame,1)
        for k,d in enumerate(dets):
            dist = []
            shape = sp(frame,d)
            face_descriptor = facerec.compute_face_descriptor(frame,shape)
            d_test = numpy.array(face_descriptor)
            for i in descriptors:
                dist_ = numpy.linalg.norm(i-d_test)
                dist.append(dist_)
            c_d = dict(zip(candidate,dist))
            cd_sorted = sorted(c_d.iteritems(),key=lambda d:d[1])
            cv2.rectangle(frame,(d.left(),d.top()),(d.right(),d.bottom()), (255,0,0))
            cv2.putText(frame,cd_sorted[0][0], (d.left(),d.top()), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
            if(cd_sorted[0][1]<0.40):
            	print("The person is:",cd_sorted[0][0])
            	rospy.set_param('recognized_person',cd_sorted[0][0])
            else:
            	print("The person is not recognized!",)
            	rospy.set_param('recognized_person',"")
        cv2.imshow('采集图像',frame)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    face_recognition()