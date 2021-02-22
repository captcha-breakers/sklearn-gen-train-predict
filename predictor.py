import pickle
import cv2
import os
import numpy as np

savfile = './sav/model.sav'
model = pickle.load(open(savfile, 'rb'))

base_dir = "./badinput/"

r,t = 0,0
for filename in os.listdir(base_dir):
    img = cv2.imread(base_dir + filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (20, 20))
    img = np.array(img).ravel()
    img = img.reshape(1, -1);
    
    # print(filename[0], " = ",  model.predict(img))
    
    if filename[0] == model.predict(img)[0]:
        r+=1
    t+=1
print(100*r/t)