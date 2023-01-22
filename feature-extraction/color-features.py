import pandas as pd
import numpy as np
import cv2

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
    df['label'] = None
    return df

def image_hue_histogram(image):
    """
    Args:
    image (numpy array): input colored  image
    Returns:
    image  features from hue histogram
    """
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
    """
    Args:
    image (numpy array): input colored  image
    Returns:
    image saturation features on HSV image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    (H,S,V) =  cv2.split(image.astype("float"))
    mean = np.mean(S)
    std = np.std(S)
    max_saturation = S.max()
    min_saturation = S.min()
    return mean,std,max_saturation,min_saturation

def image_contrast(image):
    """
    Args:
    image (numpy array): input colored  image
    Returns:
    image contrast features on HSV image
    """

    (Y,U,V) =  cv2.split(image.astype("float"))
    std = np.std(Y)
    maximum = Y.max()
    minimum = Y.min()
    if (maximum-minimum)<=0:
        return 0
    return std*1.0/(maximum-minimum)


def calculate_image_simplicity(image,c_threshold = 0.01,nchannels=3,nbins =8):
    """
    Args:
    image (numpy array): input colored image
    c_threshold (float 0-1): threshold on the maximum of the histogram value to be used in the output simplicity feature
    nchannel(int): 3 for colored images and 1 for grayscale
    nbins(int): number of bins used to calculate histogram
    Returns:
    tuple: returns 2 features representing image simplicity .
    """
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
    
    l = [filename, hsv_simple_1,hsv_simple_2, hue_hist_1,hue_hist_2,hue_hist_3,
         contrast, sat_1, sat_2, sat_3, sat_4, h_circular, v_intensity]
    
    #return list value
    return l