# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 20:24:22 2021

@author: hp
"""
from django.shortcuts import render ;
import cv2
import face_recognition
import os
import numpy as np
def home(request):
    return render(request,"home.html")
def result(request):
    KNOWN_FACES_DIR = "C:\\Users\\hp\\Desktop\\py project\\known_faces"
    known_faces = []
    known_names = []
    TOLERANCE = 0.6

    for filename in os.listdir(KNOWN_FACES_DIR):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(filename)


    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        face_locations = face_recognition.face_locations(frame)
        unknown_face_encoding = face_recognition.face_encodings(frame,face_locations)
        for face_encoding, face_loc in zip(unknown_face_encoding, face_locations):
                results = face_recognition.compare_faces(known_faces, face_encoding,TOLERANCE)
                top_left = (face_loc[1], face_loc[0])
                bottom_right = (face_loc[3], face_loc[2])
                color = (0, 0, 200)
                cv2.rectangle(frame, top_left, bottom_right, color , 4)
                match = None
                if True in np.array(results):
                        cv2.rectangle(frame, top_left, bottom_right, (0,200,0) , 4)
                        match = known_names[results.index(True)]
                        cv2.putText(frame,match[:len(match)-4], (face_loc[1], face_loc[0]-10),cv2.FONT_HERSHEY_SIMPLEX , 1 , (0,0,0) ,2 )
                else:
                    cv2.putText(frame,"Unknown", (face_loc[1], face_loc[0]-10),cv2.FONT_HERSHEY_SIMPLEX , 1 , (0,0,0) ,2 )
                         
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF== ord('q'):
            break

    cap.release()   
    return render(request,"home.html")


