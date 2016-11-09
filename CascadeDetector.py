from detector import Detector
import numpy as np
import cv2
import sys, getopt

class CascadeDetector(Detector):
    def __init__(self, cascade_filename):
        self.cascade = cv2.CascadeClassifier(cascade_filename)

    def detect(self, img):
        rects = self.cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30))
        if len(rects) == 0:
            return []
        rects[:, 2:] += rects[:, :2]
        return rects
