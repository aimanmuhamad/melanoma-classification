import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import greycomatrix, greycoprops

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