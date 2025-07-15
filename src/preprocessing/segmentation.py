import cv2
import numpy as np

def segment_lesion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    return mask, segmented
