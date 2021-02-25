from PIL import ImageFont,ImageDraw,Image
from random import random, choices
import numpy as np
import cv2
import os
from string import ascii_uppercase, ascii_lowercase, digits

for _ in range(1):
    img=np.zeros(shape=(70,185,3),dtype=np.uint8)
    img_raw=Image.fromarray(img+255)
    draw=ImageDraw.Draw(img_raw)
    font=ImageFont.truetype(font='./font/Roboto-Regular.ttf',size=40)
    my_cap = ''.join(choices(ascii_uppercase+ ascii_lowercase+ digits, k = 6)) 

    my_cap = 'ABCDEF'

    for i in range(6):
        draw.text((5+5*i+25*i,10), my_cap[i],font=font,fill=(0,0,0))

    img=np.array(img_raw)

    # Adding noise
    # thresh = 0.05
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         rdn = random()
    #         if rdn < thresh:
    #             img[i][j] = 0
    #         elif rdn > 1-thresh:
    #             img[i][j] = 255

    cv2.imwrite("./captchas/"+my_cap+".png",img)