import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from darkflow.net.build import TFNet
from tensorflow.keras import layers, models
import numpy as np
import cv2
import imutils
import string
from pprint import pformat


def firstCrop(img, predictions):
    predictions.sort(key=lambda x: x.get('confidence'))
    xtop = predictions[-1].get('topleft').get('x')
    ytop = predictions[-1].get('topleft').get('y')
    xbottom = predictions[-1].get('bottomright').get('x')
    ybottom = predictions[-1].get('bottomright').get('y')
    firstCrop = img[ytop:ybottom, xtop:xbottom]
    cv2.rectangle(img, (xtop, ytop), (xbottom, ybottom), (0, 255, 0), 3)
    return firstCrop


def secondCrop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if (len(areas) != 0):
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        secondCrop = img[y:y + h, x:x + w]
    else:
        secondCrop = img
    return secondCrop


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


class Recognizer():
    def __init__(self):
        options = {
            "pbLoad": "model_weights/yolo-plate.pb",
            "metaLoad": "model_weights/yolo-plate.meta",
        }
        self.yoloPlate = TFNet(options)

        self.characterRecognition = tf.keras.models.load_model(
            'model_weights/character_recognition.h5')

    def cnnCharRecognition(self, img):
        dictionary = {
            k: v
            for k, v in enumerate(string.digits + string.ascii_uppercase)
        }
        blackAndWhiteChar = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blackAndWhiteChar = cv2.resize(blackAndWhiteChar, (75, 100))
        image = blackAndWhiteChar.reshape((1, 100, 75, 1))
        image = image / 255.0
        new_predictions = self.characterRecognition.predict(image)
        char = np.argmax(new_predictions)
        return dictionary[char]

    def opencvReadPlate(self, img):
        charList = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh_inv = cv2.adaptiveThreshold(gray, 255,
                                           cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV, 39, 1)
        edges = auto_canny(thresh_inv)
        ctrs, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        img_area = img.shape[0] * img.shape[1]

        for i, ctr in enumerate(sorted_ctrs):
            x, y, w, h = cv2.boundingRect(ctr)
            roi_area = w * h
            non_max_sup = roi_area / img_area

            if ((non_max_sup >= 0.015) and (non_max_sup < 0.09)):
                if ((h > 1.2 * w) and (3 * w >= h)):
                    char = img[y:y + h, x:x + w]
                    charList.append(self.cnnCharRecognition(char))
                    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 0, 255), 2)
        licensePlate = "".join(charList)
        return licensePlate

    def evaluate(self, frame):
        # frame = imutils.resize(frame, 640, 480)
        # frame = imutils.rotate(frame, 270)

        licensePlate = []
        try:
            predictions = self.yoloPlate.return_predict(frame)
            firstCropImg = firstCrop(frame, predictions)
            secondCropImg = secondCrop(firstCropImg)
            secondCropImgCopy = secondCropImg.copy()
            licensePlate.append(self.opencvReadPlate(secondCropImg))
            print("OpenCV+CNN : " + licensePlate[0])
        except:
            pass

        return licensePlate
