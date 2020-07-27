# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:05:40 2020

@author: Nasim
"""
from PIL import Image
from pandas import ExcelWriter
from keras.optimizers import Adam
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
import matplotlib as plt
import pandas as pd
import numpy as np
import argparse
import locale
import glob
import Basic_CNN
import os
import cv2
import re
import pickle       
import random


def train(input_dir,input_dir_ParsedVMAF):

    def crop_center(img, crop_width, crop_height):
         width, height = img.size
         return img.crop(((width - crop_width) // 2,
                              (height - crop_height) // 2,
                              (width + crop_width) // 2,
                              (height + crop_height) // 2))
         
    def Crop_Function (img, crop_width, crop_height,i):
        width, height = img.size
        for col_i in range(0, width, crop_width):
            for row_i in range(0, height, crop_height):
                vmafs_new.append(vmafs[i])
                return img.crop((col_i, row_i, col_i + crop_width, row_i + crop_height))
       
    def resize_half(img):
         width, height = img.size
         return img.resize((width//2,height//2))
         
         
    img=[]
    vmafs=[]
    outputImage = []
    vmafs_new=[]
    imgnames=[]
    
    
    filenames = glob.glob(input_dir+'/*.png')
    for f in filenames:
        f=os.path.basename(f)
        img.append(f)
    
    df = pd.read_csv (input_dir_ParsedVMAF)
    df_list=df.values.tolist()
    for i in range(0, len(img)):
        for j in range(0, len(df_list)):
            if (img[i]==df_list[j][0]):
                imgs=img[i]
                Vmaf=df_list[j][1]
                imgnames.append(imgs)
                vmafs.append(Vmaf)
                
#    vmafs_array=np.asarray(vmafs)
    
    for i in range(len(imgnames)):
        im = Image.open(input_dir + '/' + imgnames[i] )
        width, height = im.size
        im_cropped =Crop_Function(im, 256, 256,i) 
    #   im_resized=resize_half(im)
    #   im_resized_cropped =Crop_Function(im_resized, 256, 256,i)
        im_array=img_to_array(im_cropped)
    #   im_array_resized=img_to_array(im_resized_cropped)
        norm_image =im_array / 255.0
    #   norm_image_resized =im_array_resized / 255.0
        outputImage.append(norm_image)
    #   outputImage.append(norm_image_resized)
    
    outputImage_array=np.asarray(outputImage)
    #vmafs_array = vmafs_array.reshape(-1, 1)    
    vmafs_new_array=np.asarray(vmafs_new)
    
    split = train_test_split(outputImage_array, vmafs_new_array, test_size=0.25, random_state=42)
    (trainX, testX, trainy, testy) = split
    
    trainX = np.asarray(trainX)
    testX = np.asarray(testX)
    
    model=Basic_CNN.create_cnn(256, 256, 3,regress=True) 
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
      
    print("[INFO] training model...")
    model.fit(trainX, trainy, validation_data=(testX, testy), epochs=300, batch_size=32)
    
    print("[INFO] predicting Vmaf...")
    preds = model.predict(testX)
    
    plt.pyplot.scatter(preds,testy)
    
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)    


if __name__== "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', action='store', dest='input_dir', default='./selected/' ,
                    help='Path for input images')
    parser.add_argument('-v', '--input_dir_ParsedVMAF', action='store', dest='input_dir_ParsedVMAF', default='./selected/ParsedVMAF.csv/' ,
                    help='Path for ParsedVMAF file')
    values = parser.parse_args()
    train(values.input_dir,values.input_dir_ParsedVMAF)