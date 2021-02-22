import pickle
import cv2
import os
import numpy as np

filename = './sav/model.sav'
model = pickle.load(open(filename, 'rb'))

base_dir = "./badinput/"

for filename in os.listdir(base_dir):
    img = cv2.imread(base_dir + filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (20, 20))
    img = np.array(img).ravel()
    img = img.reshape(1, -1);
    
    print(filename[0], " = ",  model.predict(img))