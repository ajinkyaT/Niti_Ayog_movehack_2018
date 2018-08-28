#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 22:52:57 2018
@author: pradeepr
Explore the sizes of all images in the indian road daatset
"""
#%%
import numpy as np
import pandas as pd
import os ,glob
from labels import labelsList
import PIL.Image as Image
#%%
image_types = ['train','val']
root_dir = '/home/pradeepr/Desktop/GOI_challenge/Niti_Ayog/AI_for_road_5/Object_detection/tensorflow_master/models/research/deeplab/datasets/cityscapes_india'
#%%
trainIds = list(labelsList.keys());
labelPixelsCount = pd.DataFrame(np.zeros((2,len(trainIds))),index=image_types ,columns=trainIds)
#%%
for i,image_type in enumerate(image_types):
   
    masks_path = os.path.join(root_dir,'gtFine',image_type,"*","*gtFine_labelTrainIds.png*")
    masks_address = glob.glob(masks_path)
    
    for j in range(len(masks_address)):
        print(j)
        img = Image.open(masks_address[j]);
        unique, counts = np.unique(img,return_counts=True)
        
        for k in range(len(unique)):
            labelPixelsCount.loc[image_type ,str(unique[k])] =  labelPixelsCount.loc[image_type,str(unique[k])] + counts[k]
            
#%%
total_no_of_training_pixels = labelPixelsCount.sum(axis=1)            
labelsFractionTrain =   labelPixelsCount.loc['train',:]/total_no_of_training_pixels['train']
trainIds = labelsFractionTrain.index
#Putting a minimum threshold on fraction so that 1/fraction will not explode
for i in range(len(labelsFractionTrain)):
    
    #Giving lower fraction i.e higher weightage for animal and boundary class
    if int(labelsFractionTrain.index[i]) == 20 or int(labelsFractionTrain.index[i]) == 24:
        labelsFractionTrain.values[i] = 0.002
    else:
        #Putting a threshold on the fraction to avoid weight explosion
        if labelsFractionTrain.values[i] < 0.01 and labelsFractionTrain.values[i] != 0:
            labelsFractionTrain.values[i] = 0.01
        
#Making Animal and boundary weightage much lessser 
labelsWeights = [1/value if value != 0 else 0 for value in labelsFractionTrain]
labelsWeights = np.array(labelsWeights)/np.sum(labelsWeights)

for i in range(len(labelsFractionTrain.index)):
    print([int(trainIds[i]), labelsWeights[i]])
    
#labelsFractionVal = labelPixelsCount.loc['val',:]/total_no_of_training_pixels['val']
#%%
if __name__ == "__main__":
    print('labelFractionVal')
#%%    