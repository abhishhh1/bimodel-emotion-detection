from facial import facial
from body import body
from openpose import makeSkelton
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import PIL
import sys
from PIL import Image
# from scipy.misc import imread
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def getActual(faceEmotion,bodyEmotion):
    bimodal={'Happy':{'Happy':'Happy', 'Sad':'Happy', 'Surprised':'Surprised', 'Angry':'Happy', 'Disgusted':'Happy', 'Neutral':'Happy', 'Fearful':'Happy'},
             'Sad':{'Happy':'Sad', 'Sad':'Sad', 'Surprised':'Sad', 'Angry':'Angry', 'Disgusted':'Disgusted' , 'Neutral':'Sad' , 'Fearful':'Fearful'},
             'Surprised':{'Happy':'Surprised' ,'Sad':'Surprised', 'Surprised':'Surprised', 'Angry':'Surprised' , 'Disgusted':'Disgusted' , 'Neutral':'Surprised' , 'Fearful':'Fearful'},
             'Angry':{'Happy':'Angry' ,'Sad':'Angry', 'Surprised':'Angry', 'Angry':'Angry' , 'Disgusted':'Disgusted' , 'Neutral':'Angry' , 'Fearful':'Fearful'},
             'Disgusted':{'Happy':'Disgusted' ,'Sad':'Sad', 'Surprised':'Disgusted', 'Angry':'Angry' , 'Disgusted':'Disgusted' , 'Neutral':'Disgusted' , 'Fearful':'Disgusted' },
             'Neutral':{'Happy':'Neutral', 'Sad':'Neutral', 'Surprised':'Neutral', 'Angry':'Neutral' , 'Disgusted':'Neutral' , 'Neutral':'Neutral', 'Fearful':'Neutral'},
             'Fearful':{'Happy':'Fearful', 'Sad':'Sad', 'Surprised':'Fearful', 'Angry':'Fearful' , 'Disgusted':'Fearful' , 'Neutral':'Fearful' , 'Fearful':'Fearful' }
            }
    
    return bimodal[faceEmotion][bodyEmotion]

def mainImage(image_path):
    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    #load model
    model.load_weights('models/model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    
    # start the stream feed  
    temp=cv2.imread(image_path)
    bodyEmotion,skelton=body(temp)
    print('Body Emotion: ', bodyEmotion)
    faceEmotion=facial(image_path)
    print('Face Emotion: ', faceEmotion)
    actual_emotion=getActual(faceEmotion,bodyEmotion)
    print('Actual Emotion: ', actual_emotion)
    
    frame=cv2.imread(image_path)
    facecasc = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
       
        label='Emotion: '+bodyEmotion
        cv2.putText(skelton, label, (x-70, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
        
        label='Face:'+faceEmotion
        cv2.putText(frame, label, (x-220, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        label='Body:'+bodyEmotion
        cv2.putText(frame, label, (x-220, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        label='Bimodal:'+actual_emotion
        cv2.putText(frame, label, (x-220, y+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite('Image-Emotion.jpg', cv2.resize(frame,(640,960),interpolation = cv2.INTER_CUBIC))
    cv2.imwrite('intermediate/Body-Emotion.jpg', cv2.resize(skelton,(640,960),interpolation = cv2.INTER_CUBIC))


def mainVideo(video_path):
    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    #load model
    model.load_weights('models/model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the stream feed
    cap = cv2.VideoCapture(video_path)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
            
        bodyEmotion,skelton=body(frame)
        facecasc = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            #cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            faceEmotion=emotion_dict[maxindex]
            actual_emotion=getActual(emotion_dict[maxindex],bodyEmotion)
            
            label='Face:'+faceEmotion
            cv2.putText(frame, label, (x-220, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            label='Body:'+bodyEmotion
            cv2.putText(frame, label, (x-220, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            label='Bimodal:'+actual_emotion
            cv2.putText(frame, label, (x-220, y+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
           

        cv2.imshow('Video', cv2.resize(frame,(640,960),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    print('Choose an option for testing:')
    print('1.Image')
    print('2.Video')
    response=int(input('Your choice number: '))
    if response==1:
        path=input('Enter input file path: ')
        mainImage(path)
    elif response==2:
        path=input('Enter input file path: ')
        mainVideo(path)
    else:
        print('Invalid choice, try again.')
