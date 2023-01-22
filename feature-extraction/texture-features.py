import pandas as pd
import numpy as np
import cv2

def create_empty_df():
    df = pd.DataFrame()
    df['filename'] = None
    df['shortrunemphasis'] = None
    df['longrunemphasis'] = None
    df['graylevelnonuniform'] = None
    df['runlengthnonuniform'] = None
    df['runpercentage'] = None
    df['lowgraylevelrun'] = None
    df['highgraylevelrun'] = None
    df['shortlowgraylevel'] = None
    df['shortrunhighgraylevel'] = None
    df['longrunlowgraylevel'] = None
    df['longrunhighgraylevel'] = None
    df['label'] = None
    return df

# Texture features
def glrlm_0(f, mask, grayLevel=5, runLength=5, skipFirstRow=True):
    degree0Matrix = np.zeros([grayLevel, runLength])
    counter = 0
    for y in range(f.shape[0]):
        for x in range(f.shape[1]):   
            nowVal = f[y][x]
            if x + 1 >= f.shape[1]:
                nextVal = None
            else:
                nextVal = f[y][x + 1]
            if nextVal != nowVal and counter == 0:
                degree0Matrix[int(nowVal)][counter] += 1
            elif nextVal == nowVal:
                counter += 1
            elif nextVal != nowVal and counter != 0:
                degree0Matrix[int(nowVal)][counter] += 1
                counter = 0
    return degree0Matrix[1:,:] if skipFirstRow else degree0Matrix

def glrlm_90(f, mask, grayLevel=5, runLength=5, skipFirstRow=True):
    degree90Matrix = np.zeros([grayLevel, runLength])
    counter = 0
    for x in range(f.shape[1]):
        for y in range(f.shape[0]):
            nowVal = f[y][x]
            if y + 1 >= f.shape[0]:
                nextVal = None
            else:
                nextVal = f[y + 1][x]
            if nextVal != nowVal and counter == 0:
                degree90Matrix[int(nowVal)][counter] += 1
            elif nextVal == nowVal:
                counter += 1
            elif nextVal != nowVal and counter != 0:
                degree90Matrix[int(nowVal)][counter] += 1
                counter = 0
    return degree90Matrix[1:,:] if skipFirstRow else degree90Matrix

def glrlm_45(f, mask, grayLevel=5, runLength=5, skipFirstRow=True):
    degree45Matrix = np.zeros([grayLevel, runLength])
    for y in range(f.shape[0]):
        counter = 0
        i_range = max(f.shape)
        for i in range(i_range):
            y1 = y - i
            if i >= f.shape[1] or y1 < 0:
                break
            else:
                nowVal = f[y1][i]
            if y1 - 1 < 0 or i + 1 >= f.shape[1]:
                nextVal = None
            else:
                nextVal = f[y1 - 1][i + 1]
            if nextVal != nowVal and counter == 0:
                degree45Matrix[int(nowVal)][counter] += 1
            elif nextVal == nowVal:
                counter += 1
            elif nextVal != nowVal and counter != 0:
                degree45Matrix[int(nowVal)][counter] += 1
                counter = 0
    for x in range(f.shape[1]):
        if x == f.shape[1] - 1:
            break
        counter = 0
        i_range = max(f.shape)
        for i in range(i_range):
            y_i = -1 - i
            x_i = -1 + i - x
            if x_i >= 0 or y_i <= -1 - f.shape[0]:
                break
            else:
                nowVal = f[y_i][x_i]
            if y_i - 1 <= -(f.shape[0] + 1) or x_i + 1 >= 0:
                nextVal = None
            else:
                nextVal = f[y_i - 1][x_i + 1]
            if nextVal != nowVal and counter == 0:
                degree45Matrix[int(nowVal)][counter] += 1
            elif nextVal == nowVal:
                counter += 1
            elif nextVal != nowVal and counter != 0:
                degree45Matrix[int(nowVal)][counter] += 1
                counter = 0
    degree45Matrix[0,1:] = 0
    return degree45Matrix[1:,:] if skipFirstRow else degree45Matrix

def glrlm_135(f, mask, grayLevel=5, runLength=5, skipFirstRow=True):
    degree135Matrix = np.zeros([grayLevel, runLength])
    for y in range(f.shape[0]):
        counter = 0
        i_range = max(f.shape)
        for i in range(i_range):
            y1 = y + i
            if y1 >= f.shape[0] or i >= f.shape[1]:
                break
            else:
                nowVal = f[y1][i]
                if y1 >= f.shape[0] - 1 or i >= f.shape[1] - 1:
                    nextVal = None
                else:
                    nextVal = f[y1 + 1][i + 1]
                if nextVal != nowVal and counter == 0:
                    degree135Matrix[int(nowVal)][counter] += 1
                elif nextVal == nowVal:
                    counter += 1
                elif nextVal != nowVal and counter != 0:
                    degree135Matrix[int(nowVal)][counter] += 1
                    counter = 0
    for x in range(f.shape[1]):
        if x == 0:
            continue
        i_range = max(f.shape)
        counter = 0
        for i in range(i_range):
            x1 = x + i
            if i >= f.shape[0] or x1 >= f.shape[1]:
                break
            else:
                nowVal = f[i][x1]
            if i >= f.shape[0] - 1 or x1 >= f.shape[1] - 1:
                nextVal = None
            else:
                nextVal = f[i + 1][x1 + 1]
            if nextVal != nowVal and counter == 0:
                degree135Matrix[int(nowVal)][counter] += 1
            elif nextVal == nowVal:
                counter += 1
            elif nextVal != nowVal and counter != 0:
                degree135Matrix[int(nowVal)][counter] += 1
                counter = 0
    degree135Matrix[0,1:] = 0
    return degree135Matrix[1:,:] if skipFirstRow else degree135Matrix

def _apply_over_degree(function, x1, x2):
    if function == np.divide:
        x2 = x2 + 1e-16
    rows, cols, nums = x1.shape
    result = np.ndarray((rows, cols, nums))
    for i in range(nums):
        result[:, :, i] = function(x1[:, :, i], x2)
        result[result == np.inf] = 0
        result[np.isnan(result)] = 0
    return result 
    
def _calculate_ij (rlmatrix):
    gray_level, run_length, _ = rlmatrix.shape
    I, J = np.ogrid[0:gray_level, 0:run_length]
    return I, J+1
    
def _calculate_s(rlmatrix):
    return np.apply_over_axes(np.sum, rlmatrix, axes=(0, 1))[0, 0]

def glrlm(f, mask, Ng=256):   
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else.
    Ng : int, optional
        Image number of gray values. The default is 256.
    Returns
    -------
    mat : numpy ndarray
        GLRL Matrices for 0, 45, 90 and 135 degrees.
    '''
    runLength = max(f.shape)
    mat0 = glrlm_0(f, mask, grayLevel=Ng, runLength=runLength)
    mat45 = glrlm_45(f, mask, grayLevel=Ng, runLength=runLength)
    mat90 = glrlm_90(f, mask, grayLevel=Ng, runLength=runLength)
    mat135 = glrlm_135(f, mask, grayLevel=Ng, runLength=runLength)            
    mat = np.dstack((mat0, mat45, mat90, mat135))      
    return mat

def glrlm_features(f, mask, Ng=256):
    if mask is None:
        mask = np.ones(f.shape)
        
    labels = ["GLRLM_ShortRunEmphasis",
              "GLRLM_LongRunEmphasis",
              "GLRLM_GrayLevelNo-Uniformity",
              "GLRLM_RunLengthNonUniformity",
              "GLRLM_RunPercentage",
              "GLRLM_LowGrayLevelRunEmphasis",
              "GLRLM_HighGrayLevelRunEmphasis",
              "GLRLM_Short owGrayLevelEmphasis",
              "GLRLM_ShortRunHighGrayLevelEmphasis",
              "GLRLM_LongRunLowGrayLevelEmphasis",
              "GLRLM_LongRunHighGrayLevelEmphasis"]
    
    rlmatrix = glrlm(f, mask, Ng)
        
    I, J = _calculate_ij(rlmatrix)
    S = _calculate_s(rlmatrix)
    G = np.apply_over_axes(np.sum, rlmatrix, axes=1)
    R = np.apply_over_axes(np.sum, rlmatrix, axes=0)
        
    features = np.zeros(11,np.double)
    features[0] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, rlmatrix, (J*J)), axes=(0, 1))[0, 0])/S).mean()
    features[1] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.multiply, rlmatrix, (J*J)), axes=(0, 1))[0, 0])/S).mean()
    features[2] = ((np.apply_over_axes(np.sum, (G*G), axes=(0, 1))[0, 0])/S).mean()
    features[3] = ((np.apply_over_axes(np.sum, (R*R), axes=(0, 1))[0, 0])/S).mean()
        
    gray_level, run_length,_ = rlmatrix.shape
    num_voxels = gray_level * run_length
    features[4] = (S/num_voxels).mean()
        
    features[5]= ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, rlmatrix, (I*I)), axes=(0, 1))[0, 0])/S).mean()
    features[6] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.multiply, rlmatrix, (I*I)), axes=(0, 1))[0, 0])/S).mean()
    features[7] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0])/S).mean()
        
    temp = _apply_over_degree(np.multiply, rlmatrix, (I*I))
    features[8] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0])/S).mean()
        
    temp = _apply_over_degree(np.multiply, rlmatrix, (J*J))
    features[9] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0])/S).mean()
    features[10] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.multiply, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0])/S).mean()
        
    return features, labels

def padImage(im, pad=2, value=0):
    TDLU=[1, 1, 1, 1]  #top, down, left, right pad
    out = im.copy()
    for _ in range(pad):
        out = cv2.copyMakeBorder(out, TDLU[0], TDLU[1], TDLU[2], TDLU[3],\
                                     cv2.BORDER_CONSTANT, None, value)
    return out

def feature_extractor(filename):
    '''
    input params: 
    filename : path of the file that we want to process

    Output params:
    l : Feature vector
    '''
    
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # Fill image with zeros outside ROI
    image = padImage(image, 2, 0)
    features = glrlm_features(image, None, Ng=256)

    shortrunemphasis = features[0][0]
    longrunemphasis = features[0][1]
    graylevelnonuniform = features[0][2]
    runlengthnonuniform = features[0][3]
    runpercentage = features[0][4]
    lowgraylevelrun = features[0][5]
    highgraylevelrun = features[0][6]
    shortlowgraylevel = features[0][7]
    shortrunhighgraylevel = features[0][8]
    longrunlowgraylevel = features[0][9]
    longrunhighgraylevel = features[0][10]

    l = [filename, shortrunemphasis, longrunemphasis, graylevelnonuniform, runlengthnonuniform, 
         runpercentage,lowgraylevelrun, highgraylevelrun, shortlowgraylevel, shortrunhighgraylevel,
         longrunlowgraylevel, longrunhighgraylevel]
    
    #return list value
    return l