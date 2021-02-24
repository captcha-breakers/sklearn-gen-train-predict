from PIL import ImageFont,ImageDraw,Image
import numpy as np
from random import choices, randint, uniform
from string import ascii_uppercase, ascii_lowercase, digits
import cv2
import os

os.system("rm -rf out; mkdir -p out")

myfonts = [ImageFont.truetype(font="./font/"+i,size=80) 
    for i in os.listdir("./font/")]

m = {}
for count in range(10000):
    img=np.zeros(shape=(100,100,3),dtype=np.uint8)
    img=Image.fromarray(img+255)
    draw=ImageDraw.Draw(img)

    char = ''.join(choices(ascii_uppercase+ ascii_lowercase+ digits, k = 1))[0]
    
    if char in m:m[char]+=1
    else:m[char] = 1
        
    try:os.mkdir("./out/"+char)
    except:pass
    
    # font=myfonts[randint(0, len(myfonts)-1)]
    font=myfonts[0]
    if char.islower(): 
        draw.text((25, -5), char,font=font,fill=(0,0,0))
    else: 
        draw.text((25, 5), char,font=font,fill=(0,0,0))

    img = img.rotate(uniform(-7, 7))
    img = np.array(img)
    img = img[5:95, 5:95]
    img = cv2.resize(img,(20,20))
    
    
    cv2.imwrite("./out/"+char+"/"+char+str(uniform(-1, 1))+".png", img)
    if count%1000 == 0:
        print(count)
print(m)