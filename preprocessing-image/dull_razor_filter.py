import cv2
import numpy as np

def dull_razor(img : np.ndarray) -> np.ndarray:
    # konversi gambar ke grayscale
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # black hat filter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)) 
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # Gaussian filter
    bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)

    # binary thresholding (mask)
    ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)

    # replace pixels of the mask
    dst = cv2.inpaint(img, mask, 6, cv2.INPAINT_TELEA)
    
    return dst
