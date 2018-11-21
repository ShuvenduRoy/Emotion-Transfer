import cv2 as cv
import glob
import numpy as np
import os
from tqdm import tqdm


face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

all_images = []

for path, subdirs, files in os.walk('data'):
    for name in files:
        all_images.append(os.path.join(path, name))


for i in tqdm(range(len(all_images))):
    image = all_images[i]

    try:
        only_name = image.split('\\')[-1]
        age = int(only_name.split('_')[0])

        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop = img[x:x + w, y:y + h]

        cv.imwrite(image, crop)

    except:
        pass

