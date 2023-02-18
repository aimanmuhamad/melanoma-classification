import pandas as pd
import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops

def create_empty_df():
    df = pd.DataFrame()
    df['filename'] = None
    df['dissimilarity_0_degree'] = None
    df['contrast_0_degree'] = None
    df['homogeneity_0_degree'] = None
    df['energy_0_degree'] = None
    df['correlation_0_degree'] = None
    df['dissimilarity_45_degree'] = None
    df['contrast_45_degree'] = None
    df['homogeneity_45_degree'] = None
    df['energy_45_degree'] = None
    df['correlation_45_degree'] = None
    df['dissimilarity_90_degree'] = None
    df['contrast_90_degree'] = None
    df['homogeneity_90_degree'] = None
    df['energy_90_degree'] = None
    df['correlation_90_degree'] = None
    df['dissimilarity_135_degree'] = None
    df['contrast_135_degree'] = None
    df['homogeneity_135_degree'] = None
    df['energy_135_degree'] = None
    df['correlation_135_degree'] = None
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
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    except:
        return filename
    
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = greycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)
    
    # Get array data
    contrast = greycoprops(glcm, 'contrast')
    dissimilarity = greycoprops(glcm, 'dissimilarity')
    homogeneity = greycoprops(glcm, 'homogeneity')
    energy = greycoprops(glcm, 'energy')
    correlation = greycoprops(glcm, 'correlation')
    
    # Ravel all data
    contrast = contrast.ravel()
    dissimilarity = dissimilarity.ravel()
    homogeneity = homogeneity.ravel()
    energy = energy.ravel()
    correlation = correlation.ravel()
    
    # 0 Degree
    dissimilarity_0_degree = dissimilarity[0]
    contrast_0_degree = contrast[0]
    homogeneity_0_degree = homogeneity[0]
    energy_0_degree = energy[0]
    correlation_0_degree = correlation[0]

    # 45 Degree
    dissimilarity_45_degree = dissimilarity[1]
    contrast_45_degree = contrast[1]
    homogeneity_45_degree = homogeneity[1]
    energy_45_degree = energy[1]
    correlation_45_degree = correlation[1]

    # 90 Degree
    dissimilarity_90_degree = dissimilarity[2]
    contrast_90_degree = contrast[2]
    homogeneity_90_degree = homogeneity[2]
    energy_90_degree = energy[2]
    correlation_90_degree = correlation[2]
    
    # 135 Degree
    dissimilarity_135_degree = dissimilarity[3]
    contrast_135_degree = contrast[3]
    homogeneity_135_degree = homogeneity[3]
    energy_135_degree = energy[3]
    correlation_135_degree = correlation[3]
    
    l = [filename, dissimilarity_0_degree, contrast_0_degree, homogeneity_0_degree, energy_0_degree, correlation_0_degree,
        dissimilarity_45_degree, contrast_45_degree, homogeneity_45_degree, energy_45_degree, correlation_45_degree,
        dissimilarity_90_degree, contrast_90_degree, homogeneity_90_degree, energy_90_degree, correlation_90_degree,
        dissimilarity_135_degree, contrast_135_degree, homogeneity_135_degree, energy_135_degree, correlation_135_degree]
    
    #return list value
    return l