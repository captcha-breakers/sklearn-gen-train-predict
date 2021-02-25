from PIL import ImageFont,ImageDraw,Image
import numpy as np
from random import choices, randint, uniform
from string import ascii_uppercase, ascii_lowercase, digits
import cv2
import os

os.system("rm -rf chars; mkdir -p chars")

# myfonts = [ImageFont.truetype(font="./font/"+i,size=80) 
#     for i in os.listdir("./font/")]
myfonts = [ImageFont.truetype(font="./font/Calibri Regular.ttf",size=100)]

all_char = ascii_uppercase+ascii_lowercase+digits

for char in all_char:
    for count in range(1000):
        img=np.zeros(shape=(100,100,3),dtype=np.uint8)
        img=Image.fromarray(img+255)
        draw=ImageDraw.Draw(img)
        
        try:os.mkdir("./chars/"+char)
        except:pass
        
        # font=myfonts[randint(0, len(myfonts)-1)]
        font=myfonts[0]
        if char.islower(): 
            draw.text((30, 0), char,font=font,fill=(0,0,0), align="center")
        else: 
            if char == 'W':
                draw.text((5, 5), char,font=font,fill=(0,0,0), align="center")
            else:
                draw.text((20, 5), char,font=font,fill=(0,0,0), align="center")

        img = img.rotate(uniform(-1, 1))
        img = np.array(img)
        img = img[5:95, 5:95]
        img = cv2.resize(img,(20,20))
        
        
        cv2.imwrite("./chars/"+char+"/"+char+str(uniform(-1, 1))+".png", img)
    print(char)