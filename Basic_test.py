# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 18:07:49 2020

@author: OmenG
"""

from PIL import Image
from pandas import ExcelWriter
from keras.optimizers import Adam
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
import matplotlib as plt
import Basic_CNN
import pandas as pd
import numpy as np
import argparse
import locale
import glob
import os
import cv2
import re
import pickle
import random


 
outputImage = []
randomList_width=[]
randomList_height=[] 
im_cropped=[]  
outputImage_half=[]   
horizontal_stack=[]
cropped_im_names=[]
cropped_im_resized_half_names=[] 
img_names=[] 

def test(input_dir,pkl_dir):
    
    with open(pkl_dir, 'rb') as file:
        pickle_model = pickle.load(file)

    filenames = glob.glob(input_dir+'/*.png')
    count = 0;
    temp1=np.empty(shape=(len(filenames),101),dtype='object')
    temp2=np.empty(shape=(len(filenames),101),dtype='object')
    for img in filenames:
        
        im = Image.open(img)
        width, height = im.size 
        randomList_width = random.sample(range(0, width-260), 100)
        randomList_height = random.sample(range(0, height-260), 100)
        
        randomList_width_half = random.sample(range(0, int((width/2)-260)), 100)
        randomList_height_half = random.sample(range(0, int((height/2)-260)), 100) 
        
        for i in range(100):
             cropped_im=im.crop((randomList_width[i], randomList_height[i], randomList_width[i] + 256, randomList_height[i] + 256))
                
             im_resized_half=im.resize((width//2,height//2))
             cropped_im_resized_half=im_resized_half.crop((randomList_width_half[i], randomList_height_half[i], randomList_width_half[i] + 256, randomList_height_half[i] + 256))

             
             im_array=img_to_array(cropped_im)
             im_array_half=img_to_array(cropped_im_resized_half)


             norm_image =im_array / 255.0
             norm_image_half =im_array_half / 255.0

             outputImage.append(norm_image)
             outputImage_half.append(norm_image_half)
             

        
        outputImage_array=np.asarray(outputImage)
        outputImage_half_array=np.asarray(outputImage_half)
    
        testX=outputImage_array 
        testX_half=outputImage_half_array 
    
    
        preds = pickle_model.predict(testX)
        preds_half = pickle_model.predict(testX_half)


        
        temp1[count,0]= os.path.basename(img)
        temp1[count,1:101]=np.transpose(preds[:100])  
        temp1_df=pd.DataFrame(temp1)
        np.delete(preds,0)
        
        temp2[count,0]= os.path.basename(img)
        temp2[count,1:101]=np.transpose(preds_half[:100])  
        temp2_df=pd.DataFrame(temp2)
        np.delete(preds_half,0)
        
        outputImage.clear()
        outputImage_half.clear()
        count = count + 1;

    with pd.ExcelWriter('output.xlsx') as writer:  
        temp1_df.to_excel(writer, sheet_name='Sheet_name_1')
        temp2_df.to_excel(writer, sheet_name='Sheet_name_2')

    return


if __name__== "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', action='store', dest='input_dir', default='./test/' ,
                    help='Path for input images')
    parser.add_argument('-p', '--pkl_dir', action='store', dest='pkl_dir', default='./pickle_model.pkl/' ,
                    help='Path for pickel file')
    values = parser.parse_args()
    test(values.input_dir,values.pkl_dir)