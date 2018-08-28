from __future__ import division

import os
from tqdm import tqdm
from keras.models import model_from_json
import numpy as np
from imageio import imwrite, imread

import cv2
#%%
def read_model(cross='_rgb_less_class_'):
    json_name = 'architecture_64_100_' + cross + '.json'
    weight_name = 'model_weights_64_100_' + cross + '.h5'
    model = model_from_json(open(os.path.join('../cache', json_name)).read())
    model.load_weights(os.path.join('../cache', weight_name))
    return model

#%%
path = '../year_wise_data/'

model = read_model()
places_name = os.listdir(path)  

num_channels = 3
num_mask_channels = 10
threshold = 0.3

classes = ['Buildings','Misc','roads', 'track', 'trees', 'crops', 'waterways', 'standing_water', 'large_vehicle', 'small_vehicle']
#color ={1:( 70, 70, 70), 2:(102,102,156), 3:(128, 64,128),4:(150,100,100), 5:(16,150,20), 6:(107,142, 35), 
#                     7:(70,60,180), 8:(70,60,180), 9:(  0, 80,100), 10:(100,80,10)}
class_ids = [1, 3, 6, 7, 9, 10]  # for training purpose
color = {1:(255,153,0), 2:(153,102,0), 3:(153,255,102),4:(51,204,255), 5:(255,0,255), 6:(255,0,255)}

class_threshold = {1:0.9, 3:0.3, 6:0.4, 7:0.8,  9:0.3, 10:0.3}
background_clr = (102,153,153)    
#%%
def make_prediction_cropped(model, X_train, initial_size=(572, 572), final_size=(388, 388), num_channels=19, num_masks=10):
    shift = int((initial_size[0] - final_size[0]) / 2)

    height = X_train.shape[1]
    width = X_train.shape[2]

    if height % final_size[1] == 0:
        num_h_tiles = int(height / final_size[1])
    else:
        num_h_tiles = int(height / final_size[1]) + 1

    if width % final_size[1] == 0:
        num_w_tiles = int(width / final_size[1])
    else:
        num_w_tiles = int(width / final_size[1]) + 1

    rounded_height = num_h_tiles * final_size[0]
    rounded_width = num_w_tiles * final_size[0]

    padded_height = rounded_height + 2 * shift
    padded_width = rounded_width + 2 * shift

    padded = np.zeros((num_channels, padded_height, padded_width))

    padded[:, shift:shift + height, shift: shift + width] = X_train

    # add mirror reflections to the padded areas
    up = padded[:, shift:2 * shift, shift:-shift][:, ::-1]
    padded[:, :shift, shift:-shift] = up

    lag = padded.shape[1] - height - shift
    bottom = padded[:, height + shift - lag:shift + height, shift:-shift][:, ::-1]
    padded[:, height + shift:, shift:-shift] = bottom

    left = padded[:, :, shift:2 * shift][:, :, ::-1]
    padded[:, :, :shift] = left

    lag = padded.shape[2] - width - shift
    right = padded[:, :, width + shift - lag:shift + width][:, :, ::-1]

    padded[:, :, width + shift:] = right

    h_start = range(0, padded_height, final_size[0])[:-1]
    assert len(h_start) == num_h_tiles

    w_start = range(0, padded_width, final_size[0])[:-1]
    assert len(w_start) == num_w_tiles

    temp = []
    for h in h_start:
        for w in w_start:
            temp += [padded[:, h:h + initial_size[0], w:w + initial_size[0]]]

    prediction = model.predict(np.array(temp))

    predicted_mask = np.zeros((num_masks, rounded_height, rounded_width))

    for j_h, h in enumerate(h_start):
         for j_w, w in enumerate(w_start):
             i = len(w_start) * j_h + j_w
             predicted_mask[:, h: h + final_size[0], w: w + final_size[0]] = prediction[i, :, 16:16+final_size[0], 16:16+final_size[0]]

    return predicted_mask[:, :height, :width]  

#%%
def rgb_mask(predicted_mask, mask_to_draw):
    mask_r = mask_to_draw[:,:,0]
    mask_g = mask_to_draw[:,:,1]
    mask_b = mask_to_draw[:,:,2]
    for i in range(num_mask_channels):
        if not (i==7)+(i==8) ==1:
#            mask_r[predicted_mask[i] < class_threshold[i+1]] = background_clr[0]
#            mask_g[predicted_mask[i] < class_threshold[i+1]] = background_clr[1] 
#            mask_b[predicted_mask[i] < class_threshold[i+1]] = background_clr[2]
            mask_r[predicted_mask[i] >= class_threshold[i+1]] = color[i+1][0]
            mask_g[predicted_mask[i] >= class_threshold[i+1]] = color[i+1][1] 
            mask_b[predicted_mask[i] >= class_threshold[i+1]] = color[i+1][2]
    mask_r = np.expand_dims(mask_r, 2)
    mask_g = np.expand_dims(mask_g, 2)
    mask_b = np.expand_dims(mask_b, 2)
    mask_to_draw = np.concatenate([mask_r, mask_g, mask_b], axis=2)
    return mask_to_draw
    
#%%    
for name in places_name:
    save_path ='../test_rgb/'+name
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for test_image in os.listdir(os.path.join(path, name)):
         image = imread(path+'/'+name+'/'+test_image)
         image_transpose = np.transpose(image,(2,0,1))
         H = image_transpose.shape[1]
         W = image_transpose.shape[2]
         
         predicted_mask = make_prediction_cropped(model, image_transpose, initial_size=(128, 128),
                                                  final_size=(128-32, 128-32),
                                                  num_masks=num_mask_channels, num_channels=num_channels)
    
         mask_to_draw = np.zeros((H,W,3),np.uint8)
         mask_to_draw = rgb_mask(predicted_mask, mask_to_draw)
#         for i in range(num_mask_channels):
#               mask_to_draw[predicted_mask[i]>=class_threshold[i+1]] = color[i+1]
               #mask_to_draw[predicted_mask[i]<threshold] = (255,188,64)   
         image_mask = cv2.addWeighted(image, 0.6, mask_to_draw, 0.4, 0)
         imwrite(save_path+'/{}_mask.png'.format(test_image), mask_to_draw)
         imwrite(save_path+'/{}image.png'.format(test_image), image)
         imwrite(save_path+'/{}_mask_image.png'.format(test_image), image_mask)
    
#%%
    
