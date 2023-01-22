import pandas as pd
import numpy as np
import cv2
from scipy.stats import entropy, pearsonr
from minepy import MINE

def create_empty_df():
    df = pd.DataFrame()
    df['filename'] = None
    df['hsv_simple_1'] = None
    df['hsv_simple_2'] = None
    df['hue_hist_1'] = None
    df['hue_hist_2'] = None
    df['hue_hist_3'] = None
    df['contrast'] = None
    df['sat_1'] = None
    df['sat_2'] = None
    df['sat_3'] = None
    df['sat_4'] = None
    df['h_circular'] = None
    df['v_intensity'] = None
    df['entropy_hue'] = None
    df['entropy_saturation'] = None
    df['entropy_value'] = None
    df['rms_contrast_hue'] = None
    df['rms_contrast_sat'] = None
    df['rms_contrast_val'] = None
    df['mad_hue'] = None
    df['mad_saturation'] = None
    df['mad_val'] = None
    df['mi_saturation_value'] = None
    df['mi_hue_value'] = None
    df['mi_hue_saturation'] = None
    df['corr_hue_saturation'] = None
    df['corr_hue_value'] = None
    df['corr_saturation_value'] = None
    df['label'] = None
    return df

def image_hue_histogram(image):
    image =  cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    (H,S,V) =  cv2.split(image.astype("float"))
    hist = cv2.calcHist(image, [0], None,[20],[0,256])
    c_threshold = 0.01
    maximum = hist.max()
    feature_1 = np.sum([1 if hist[i]>=(c_threshold*maximum) else 0 for i in range(20)])
    max_2 = -1

    for i in range(20):
        if hist[i]==maximum:
            continue
        if hist[i]>max_2:
            max_2 = hist[i]
        
    feature_2 = maximum-max_2

    return feature_1,feature_2[0],np.std(H)

def image_saturation(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    (H,S,V) =  cv2.split(image.astype("float"))
    mean = np.mean(S)
    std = np.std(S)
    max_saturation = S.max()
    min_saturation = S.min()
    return mean,std,max_saturation,min_saturation

def image_contrast(image):
    (Y,U,V) =  cv2.split(image.astype("float"))
    std = np.std(Y)
    maximum = Y.max()
    minimum = Y.min()
    if (maximum-minimum)<=0:
        return 0
    return std*1.0/(maximum-minimum)


def calculate_image_simplicity(image,c_threshold = 0.01,nchannels=3,nbins =8):
    feature_1 =0
    max_bin = -1
    max_channel = -1
    bin_index = -1
    for channel in  range(nchannels):
        hist = cv2.calcHist(image, [channel], None,[nbins],[0,256])
        maximum = hist.max()
        feature_1 += np.sum([1 if hist[i]>=(c_threshold*maximum) else 0 for i in range(8)])

        if max_bin<maximum:
            max_bin = maximum
            max_channel = channel
            bin_index = np.where(hist == max_bin)[0]

    feature_2 = max_bin *100.0 /  image.flatten().shape[0]
    return feature_1,feature_2

def image_hsv_simplicity(image):
    return calculate_image_simplicity(image,0.05,1,20)

def entropy_value(img_hsv):
    hue_channel = img_hsv[:,:,0]
    saturation_channel = img_hsv[:,:,1]
    value_channel = img_hsv[:,:,2]
    entropy_hue = entropy(hue_channel.ravel())
    entropy_saturation = entropy(saturation_channel.ravel())
    entropy_value = entropy(value_channel.ravel())

    return entropy_hue, entropy_saturation, entropy_value

def rms_contrast_value(img_hsv):
    hue_channel = img_hsv[:,:,0]
    saturation_channel = img_hsv[:,:,1]
    value_channel = img_hsv[:,:,2]
    min_hue, max_hue = np.min(hue_channel), np.max(hue_channel)
    rms_contrast_hue = max_hue - min_hue
    min_saturation, max_saturation = np.min(saturation_channel), np.max(saturation_channel)
    rms_contrast_sat = max_saturation - min_saturation
    min_value, max_value = np.min(value_channel), np.max(value_channel)
    rms_contrast_val = max_value - min_value

    return rms_contrast_hue, rms_contrast_sat, rms_contrast_val

def mad_value(img_hsv):
    hue_channel = img_hsv[:,:,0]
    saturation_channel = img_hsv[:,:,1]
    value_channel = img_hsv[:,:,2]
    mean_hue = np.mean(hue_channel)
    mean_saturation = np.mean(saturation_channel)
    mean_value = np.mean(value_channel)

    mad_hue = np.mean(np.abs(hue_channel - mean_hue))
    mad_saturation = np.mean(np.abs(saturation_channel - mean_saturation))
    mad_val = np.mean(np.abs(value_channel - mean_value))

    return mad_hue, mad_saturation, mad_val

def pearson_corr_value(img_hsv):
    hue_channel = img_hsv[:,:,0]
    saturation_channel = img_hsv[:,:,1]
    value_channel = img_hsv[:,:,2]

    corr_hue_saturation, _ = pearsonr(hue_channel.ravel(), saturation_channel.ravel())
    corr_hue_value, _ = pearsonr(hue_channel.ravel(), value_channel.ravel())
    corr_saturation_value, _ = pearsonr(saturation_channel.ravel(), value_channel.ravel())

    return corr_hue_saturation, corr_hue_value, corr_saturation_value

def mutual_information_value(img_hsv):
    hue_channel = img_hsv[:,:,0]
    saturation_channel = img_hsv[:,:,1]
    value_channel = img_hsv[:,:,2]
    
    mine = MINE()
    mine.compute_score(hue_channel, saturation_channel)
    mi_hue_saturation = mine.mic()

    mine = MINE()
    mine.compute_score(hue_channel, value_channel)
    mi_hue_value = mine.mic()
  
    mine = MINE()
    mine.compute_score(saturation_channel, value_channel)
    mi_saturation_value = mine.mic()

    return mi_saturation_value, mi_hue_value, mi_hue_saturation

def compute_circular(channel_image):
    A = np.cos(channel_image).sum()
    B = np.sin(channel_image).sum()
    R = 1 - np.sqrt(A ** 2 + B ** 2) / (channel_image.shape[0] * channel_image.shape[1])
    return R

def compute_hsv_statics(hsv_img):
    h_circular = compute_circular(hsv_img[0])
    v_intensity = np.sqrt((hsv_img[-1] ** 2).mean())

    return h_circular, v_intensity

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
    
    current_frame = main_img
    filtered_image = closing/255

    #Elementwise Multiplication of range bounded filtered_image with current_frame
    current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 0] = np.multiply(current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 0], filtered_image) #B channel
    current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 1] = np.multiply(current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 1], filtered_image) #G channel
    current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 2] = np.multiply(current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 2], filtered_image) #R channel

    img = current_frame
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    image = img
        
    contrast = image_contrast(image)
    sat_1,sat_2,sat_3,sat_4 = image_saturation(image)
    hue_hist_1,hue_hist_2,hue_hist_3 = image_hue_histogram(image)
    hsv_simple_1,hsv_simple_2 = image_hsv_simplicity(image)
    
    h_circular = compute_circular(image[0])
    v_intensity = np.sqrt((image[-1] ** 2).mean())
    
    entropy_hue, entropy_saturation, entropy_value = entropy_value(image)
    rms_contrast_hue, rms_contrast_sat, rms_contrast_val = rms_contrast_value(image)
    mad_hue, mad_saturation, mad_val = mad_value(image)
    corr_hue_saturation, corr_hue_value, corr_saturation_value = pearson_corr_value(image)
    mi_saturation_value, mi_hue_value, mi_hue_saturation = mutual_information_value(image)
    
    l = [filename, hsv_simple_1, hsv_simple_2, hue_hist_1,hue_hist_2,hue_hist_3,
         contrast, sat_1, sat_2, sat_3, sat_4, h_circular, v_intensity, 
         entropy_hue, entropy_saturation, entropy_value,
         rms_contrast_hue, rms_contrast_sat, rms_contrast_val,
         mad_hue, mad_saturation, mad_val,
         mi_saturation_value, mi_hue_value, mi_hue_saturation,
         corr_hue_saturation, corr_hue_value, corr_saturation_value
         ]
    
    #return list value
    return l