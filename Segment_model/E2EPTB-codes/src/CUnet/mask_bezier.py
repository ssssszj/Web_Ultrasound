"""Mask red annotation from medical ultrasound images"""

# Import necessary packages
import cv2
import numpy as np
import os
import shutil
from PIL import Image


# define PATH for ROI and Mask directory
imgPath = '../../data/inpaintImages/'
pathROI = '../../data/Bezier/'
pathMask = '../../data/Beziermask'

if not os.path.exists(pathMask):
    os.makedirs(pathMask)

# Read all images from cervix directory
images = os.listdir(pathROI)
print("Our dataset contains {} images!".format(len(images)))

for img in images:
    img_path = os.path.join(pathROI, img)
    image = cv2.imread(img_path)
    # Transition to the HSV color space
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    except cv2.error:
        continue
    # Define range of min and max H, S, V
    lower_red = np.array(([0, 100, 100]), dtype=np.uint8)
    upper_red = np.array(([20, 255, 255]), dtype=np.uint8)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask = mask_red

    cv2.imwrite(os.path.join(pathMask, img), mask)

    #os.remove(os.path.join(pathROI, img))
    name = img.replace('.png', '_inpaint.png')
    try:
        shutil.copyfile(os.path.join(imgPath, name), img_path)
        im = Image.open(img_path)
        width, height = im.size
        offset = 450
        # im.crop((width//2 - offset//2, height//2 - offset//2, width//2 + offset//2, height//2 + offset//2)).save(img_path)
        im.save(img_path)
        
    except FileNotFoundError:
        continue