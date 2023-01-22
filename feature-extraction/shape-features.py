import pandas as pd
import cv2
import numpy as np

def create_empty_df():
    df = pd.DataFrame()
    df['filename'] = None
    df['area'] = None
    df['filledArea'] = None
    df['perimeter'] = None
    df['solidity'] = None
    df['equi_diameter'] = None
    df['aspect_ratio'] = None
    df['orientation'] = None
    df['major_axis_length'] = None
    df['minor_axis_length'] = None
    df['eccentricity'] = None
    df['extent'] = None
    df['convex_area'] = None
    df['label'] = None
    return df

def feature_extractor(filename):
    '''
    input params: 
    filename : path of the file that we want to process

    Output params:
    l : Feature vector
    '''
    try:
        main_img = cv2.imread(filename)
        img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
    except:
        print(filename)

    #Preprocessing
    gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gs, (25,25),0)
    ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((25,25),np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
    
    # Shape features
    contours, _ = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    M = cv2.moments(cnt)
    area = cv2.contourArea(cnt)
    if area==0:
        print(filename)
    
    perimeter = cv2.arcLength(cnt,True)
    convex_hull = cv2.convexHull(cnt)
    convex_area = cv2.contourArea(convex_hull)
    try:
        solidity = float(area)/convex_area
    except:
        solidity = 0
    x,y,w,h = cv2.boundingRect(cnt)
    rect_area = w*h
    extent = float(area)/rect_area
    equi_diameter = np.sqrt(4*area/np.pi)
    aspect_ratio = float(w)/h
    filledImage = np.zeros(img.shape[0:2],np.uint8)
    cv2.drawContours(filledImage,[cnt],0,255,-1)
    # area of filled image
    filledArea = cv2.countNonZero(filledImage)

    try :
        ellipse = cv2.fitEllipse(cnt)
        (center, axes, orientation) = ellipse
        major_axis_length = max(axes)
        minor_axis_length = min(axes)
        eccentricity = np.sqrt(1-(minor_axis_length/major_axis_length)**2)
        orientation = int(orientation)
    except :
        major_axis_length = 0
        minor_axis_length = 0
        eccentricity = 0
        orientation = 0

    l = [filename, area, filledArea, perimeter, solidity, equi_diameter, aspect_ratio, orientation, 
         major_axis_length, minor_axis_length,
         eccentricity, extent, convex_area]
    
    #return list value
    return l