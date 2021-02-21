#Importing Libraries :
from sklearn import svm #SupportVectorMachine
import numpy as np
import os
import cv2
from random import randint

base_dir = "./Bmp/"
is_char = {1 : '0',2 : '1',3 : '2',4 : '3',5 : '4',6 : '5',7 : '6',8 : '7',9 : '8',10 : '9',11 : 'A',12 : 'B',13 : 'C',14 : 'D',15 : 'E',16 : 'F',17 : 'G',18 : 'H',19 : 'I',20 : 'J',21 : 'K',22 : 'L',23 : 'M',24 : 'N',25 : 'O',26 : 'P',27 : 'Q',28 : 'R',29 : 'S',30 : 'T',31 : 'U',32 : 'V',33 : 'W',34 : 'X',35 : 'Y',36 : 'Z',37 : 'a',38 : 'b',39 : 'c',40 : 'd',41 : 'e',42 : 'f',43 : 'g',44 : 'h',45 : 'i',46 : 'j',47 : 'k',48 : 'l',49 : 'm',50 : 'n',51 : 'o',52 : 'p',53 : 'q',54 : 'r',55 : 's',56 : 't',57 : 'u',58 : 'v',59 : 'w',60 : 'x',61 : 'y',62 : 'z'}

X = []
y = []
print("Reading images...")
for i in range(5):
    each_dir = "Sample"+str(i+1).zfill(3)+"/"
    files = os.listdir(base_dir+each_dir)
    for file in files:
        img = cv2.imread(base_dir+each_dir+file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 100)) 
        
        img = np.array(img).ravel()
        flat_img = img.reshape(-1) 
        
        X.append(flat_img)
        y.append(is_char[i+1])
print("Reading images completed.")
    
print("Fit started...")
clf = svm.SVC(C=1, kernel="linear") #SVM Classifier
clf.fit(X, y)
print("Fit started completed.")

for i in X:clf.predict(i.reshape(1, -1))
print(clf.score(X, y))


import pickle
os.system("rm -rf sav; mkdir -p sav")
filename = './sav/model.sav'
pickle.dump(clf, open(filename, 'wb'))