"""Importing libraries"""
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
import cv2
import os

base_dir = "./D-captchas-simple-uppercase/data/"  # Input directory to predict from
r, t = 0, 0
for filename in os.listdir(base_dir):
    """Segmentation"""
    image = cv2.imread(base_dir+filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    cnts, new = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    new_image = cv2.drawContours(image, cnts, -1, (0, 255, 0), 1)
    NumberPlateCnt = None
    captcha = np.invert(thresh)
    labelled_captcha = measure.label(captcha)

    character_dimensions = (
        0.2*captcha.shape[0], 0.9*captcha.shape[0],
        0.05*captcha.shape[1], 0.5*captcha.shape[1]
    )
    min_height, max_height, min_width, max_width = character_dimensions

    characters = []
    column_list = []
    # fig, ax1 = plt.subplots(1)
    # ax1.imshow(captcha, cmap="gray")

    for regions in regionprops(labelled_captcha):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0

        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            roi = captcha[y0:y1, x0:x1]
            # draw a red bordered rectangle over the character.
            # rect_border = patches.Rectangle(
            #     (x0-2, y0-2), x1 - x0 + 3, y1 - y0 + 3,
            #     edgecolor="red",linewidth=2, fill=False
            # )
            # ax1.add_patch(rect_border)
            characters.append((x0, roi))
            # print(regions.bbox)

    # plt.show()

    """PredictCharacters"""
    model_file = "./sav/model.sav"
    model = pickle.load(open(model_file, 'rb'))

    res = []
    for i in characters:
        x_p, y_p = 3, 3
        img = cv2.copyMakeBorder(i[1], x_p, x_p, y_p, y_p, cv2.BORDER_CONSTANT)
        img = cv2.resize(img, (20, 20))

        result = model.predict(img.reshape(1, -1))
        res.append((i[0], result))

    res.sort(key=lambda x: x[0])

    predicted_captcha = ''
    for i in res:
        predicted_captcha += str(i[1][0])

    if filename[:6] == predicted_captcha:
        r += 1
    t += 1
    os.system("clear")
    print("Total images processed:", t)
    print("Accuracy: ", 100*r/t)
    print(filename[:6], " : ", predicted_captcha, " : ", filename[:6] == predicted_captcha)
