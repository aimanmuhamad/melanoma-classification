import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import greycomatrix, greycoprops

def create_empty_df():
    df = pd.DataFrame()
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
    df['red_mean'] = None
    df['green_mean'] = None
    df['blue_mean'] = None
    df['f1'] = None
    df['f2'] = None
    df['red_std'] = None
    df['green_std'] = None
    df['blue_std'] = None
    df['f4'] = None
    df['f5'] = None
    df['f6'] = None
    df['f7'] = None
    df['f8'] = None
    df['f9'] = None
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
        return "Invalid"
    
    #Preprocessing
    gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gs, (25,25),0)
    ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((25,25),np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

    #Shape features
    contours, _ = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    M = cv2.moments(cnt)
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    if area==0:
        return "Invalid"

    perimeter = cv2.arcLength(cnt,True)
    solidity = float(area)/hull_area
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
      print("Failed")

    current_frame = main_img
    filtered_image = closing/255

    #Elementwise Multiplication of range bounded filtered_image with current_frame
    current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 0] = np.multiply(current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 0], filtered_image) #B channel
    current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 1] = np.multiply(current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 1], filtered_image) #G channel
    current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 2] = np.multiply(current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 2], filtered_image) #R channel

    img = current_frame

    #Color features
    red_channel = img[:,:,0]
    green_channel = img[:,:,1] #show the intensities of green channe
    blue_channel = img[:,:,2]

    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)
    #standard deviation for colour feature from the image.    
    red_std = np.std(red_channel)
    green_std = np.std(green_channel)
    blue_std = np.std(blue_channel)
    
    #amt.of green color in the image
    gr = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    boundaries = [([30,0,0],[70,255,255])]
    for (lower, upper) in boundaries:
        mask = cv2.inRange(gr, (36, 0, 0), (70, 255,255))
        ratio_green = cv2.countNonZero(mask)/(img.size/3)
        f1=np.round(ratio_green, 2)
    #amt. of non green part of the image   
    f2=1-f1

    #Texture features using grey level cooccurance matrix
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g=greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])

    #with the help of glcm find the contrast
    contrast = greycoprops(g, 'contrast')
    f4=contrast[0][0]+contrast[0][1]+contrast[0][2]+contrast[0][3]
    #[0][3] represent no. of times grey level 3 appears at the right of 0

    #with the help of glcm find the dissimilarity 
    dissimilarity = greycoprops(g, prop='dissimilarity')
    f5=dissimilarity[0][0]+dissimilarity[0][1]+dissimilarity[0][2]+dissimilarity[0][3]

    #with the help of glcm find the homogeneity
    homogeneity = greycoprops(g, prop='homogeneity')
    f6=homogeneity[0][0]+homogeneity[0][1]+homogeneity[0][2]+homogeneity[0][3]

    #with the help of glcm find the energy
    energy = greycoprops(g, prop='energy')
    f7=energy[0][0]+energy[0][1]+energy[0][2]+energy[0][3]

    #with the help of glcm find the correlation
    correlation = greycoprops(g,prop= 'correlation')
    f8=correlation[0][0]+correlation[0][1]+correlation[0][2]+correlation[0][3]

    #with the help of glcm find the ASM
    asm = greycoprops(g, prop= 'ASM')
    f9 = asm[0][0] + asm[0][1] + asm[0][2] + asm[0][3]

    l = [area, filledArea, perimeter, solidity, equi_diameter, aspect_ratio, orientation, major_axis_length, minor_axis_length,
         eccentricity, extent, red_mean, green_mean, blue_mean, 
         f1, f2, red_std, green_std, blue_std, f4, f5, f6, f7, f8, f9]
    
    #return list value
    return l

def process_folder(folderpath,df_f,label_f):
    '''
    input params:
    folderpath : Path of the folder that we want to process
    df_f = dataframe for specific disease
    label_f : label corresponding to the specific disease

    Output params:
    df_f = Dataframe consisting processed vectors
    '''
    imagelist = os.listdir(folderpath)  # stores all the imagepaths in the python list
    for image in imagelist:
        imagepath = os.path.join(folderpath, image)
        im_feature = feature_extractor(imagepath) 
        if im_feature == "Invalid":
            continue
        im_feature.append(label_f)  # appending label to feature vector
        df_f.loc[len(df_f)] = im_feature 
        if len(df_f)%500 ==0:
            print(len(df_f))

    return df_f

def process_plant(folderpaths, labels, savepath):
    '''
    input params:
    folderpaths : List of the folderpaths for specific Plant
    labels : List of labels 
    savepath : Path to export datasheet

    Output params:
    None
    '''
    datasheet = create_empty_df()
    for i in range(len(folderpaths)):
        datasheet = process_folder(folderpaths[i],datasheet,labels[i])

    datasheet.to_excel(savepath)

    return None
