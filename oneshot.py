import sys
import numpy as np
# import pandas as pd
import PIL
# from scipy.misc import imread
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import cv2
import time

import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K

from sklearn.utils import shuffle

import numpy.random as rng

def loadimgs(path):
    '''
    path => Path of train directory or test directory
    '''
    X=[]

    for classes in os.listdir(path):
        #print("loading class: " + classes)
        class_path = os.path.join(path,classes)
        
        for img_dir in os.listdir(class_path):
            category_images=[]
            img_path = os.path.join(class_path, img_dir)
            
            for filename in os.listdir(img_path):
                image_name = os.path.join(img_path, filename)
                image = imread(image_name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (105, 105))
                
                category_images.append(image)

            try:
                X.append(np.stack(category_images))
            # edge case  - last one
            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)

    X = np.stack(X)
    return X
    
def initialize_weights(shape, name=None, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    value = np.random.normal(loc = 0.0, scale = 1e-2, size = shape)
    return K.variable(value, name=name, dtype=dtype)

def initialize_bias(shape, name=None, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    value = np.random.normal(loc = 0.5, scale = 1e-2, size = shape)
    return K.variable(value, name=name, dtype=dtype)


def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu',kernel_initializer=initialize_weights,bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=initialize_weights,bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=initialize_weights,bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',kernel_regularizer=l2(1e-3),kernel_initializer=initialize_weights, bias_initializer=initialize_bias))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net


def make_oneshot_task(img, Xtrain):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    X = Xtrain
    categories = [0, 1, 2, 3, 4, 5, 6]

    n_classes, n_examples, w, h = X.shape

    test_image = np.asarray([img]*n_classes*n_examples).reshape(n_classes*n_examples, w, h, 1)

    support_set = []#X[categories, indices, :, :]
    for i in categories:
        for j in range(n_examples):
            support_set.append(X[i, j, :, :])
            
    support_set = np.asarray(support_set)
    
    support_set = support_set.reshape(n_classes*n_examples, w, h, 1)
    
    targets = {0: "Surprised", 1: "Sad", 2: "Happy", 3: "Angry", 4: "Fearful", 5: "Disgusted", 6: "Neutral"}

    pairs = [test_image, support_set]

    return pairs, targets, categories


def test_oneshot(model, img, Xtrain):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    
    inputs, targets, cat = make_oneshot_task(img, Xtrain)
    probs = model.predict(inputs)
    n_correct = np.argmax(probs)
        
    percent_correct = targets[n_correct//10]
    
    #label ='Body: '+percent_correct
    
    #cv2.putText(img, label, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #cv2.imwrite('intermediate/Body-Output.jpg',img)
    
    return percent_correct
    

def oneshot(image):
    X=loadimgs("train")
    model = get_siamese_model((105, 105, 1))
    #print(model.summary())

    optimizer = Adam(lr = 0.00006)
    model.compile(loss="binary_crossentropy",optimizer=optimizer)
    
    model_path = 'weights'
    model.load_weights(os.path.join(model_path, "weights.1600.h5"))
    #image = imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (105, 105))
    #cv2.imwrite('input-img.jpg',image)
    return test_oneshot(model, image, X)
    
#print(oneshot('disgust.jpg'))

