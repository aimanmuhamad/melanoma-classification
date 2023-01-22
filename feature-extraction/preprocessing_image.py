from module import *

global_folder = YOUR_FOLDER

folderpaths = [global_folder + 'NotMelanoma',
              global_folder + 'Melanoma'
              ]
'''
input params:
folderpaths : List of the folderpaths for specific Plant
labels : List of labels 
savepath : Path to export datasheet
'''
labels = [0,1]
savepath = YOUR_FOLDER
process_plant(folderpaths, labels, savepath)