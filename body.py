from openpose import makeSkelton
from oneshot import oneshot

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

def body(image_path):
    skelton=makeSkelton(image_path)
    #cv2.imwrite('intermediate_images/boyd-skelton.jpg',skelton)
    emotion=oneshot(skelton)
    return emotion,skelton

