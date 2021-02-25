from sklearn import svm
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
from string import ascii_uppercase, ascii_lowercase, digits
import os
import cv2
import random
random.seed(42)

base_dir = "./chars/"

images = []
# def showImage(img):
    # cv2.imwrite("./imd/"+str(random())+".png", img)
    # cv2.imshow("imageShow", img)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()

print("Reading images...")
for f in os.listdir(base_dir):
# for f in ascii_uppercase[:6]:
    for file in os.listdir(base_dir+f)[:1000]:
        img = cv2.imread(base_dir+f+"/"+file, cv2.IMREAD_GRAYSCALE)
        img = np.invert(img)
        # _, thresh = cv2.threshold(img, 120, 255 , cv2.THRESH_BINARY)
        
        # showImage(img)
        img = cv2.resize(img, (20,20))
        
        img = np.array(img).ravel()
        img = img.reshape(-1) 
        
        images.append((img, file[0]))
    print(f)

random.shuffle(images)
print("Reading images completed.")

def cross_validation(model, num_of_fold, train_data, train_label):
    # this uses the concept of cross validation to measure the accuracy
    # of a model, the num_of_fold determines the type of validation
    # e.g if num_of_fold is 4, then we are performing a 4-fold cross validation
    # it will divide the dataset into 4 and use 1/4 of it for testing
    # and the remaining 3/4 for the training
    accuracy_result = cross_val_score(
        model, train_data, train_label,cv=num_of_fold)
    print(str(num_of_fold), "-fold cross validation result: ", accuracy_result * 100)

print("Fit started...")
X,y = [],[]
for i in images:X.append(i[0]),y.append(i[1])
clf = svm.SVC(C=1, kernel="linear") #SVM Classifier
clf.fit(X, y)
print("Fit started completed.")

os.system("rm -rf sav; mkdir -p sav")
filename = './sav/model.sav'
pickle.dump(clf, open(filename, 'wb'))
print("SAV generation complete.")

cross_validation(clf, 5, X, y)
print("Cross validation complete.")

for i in X:clf.predict(i.reshape(1, -1))
print(clf.score(X, y))
print("Score complete.")