from __future__ import division

import os
from tqdm import tqdm
import pandas as pd
import extra_functions
import shapely.geometry
import tifffile as tiff
from keras.models import model_from_json
import numpy as np
from imageio import imwrite, imread
import cv2
#%%
def read_model(cross='_all_channel_first_'):
    json_name = 'architecture_16_50_' + cross + '.json'
    weight_name = 'model_weights_16_50_' + cross + '.h5'
    model = model_from_json(open(os.path.join('../cache', json_name)).read())
    model.load_weights(os.path.join('../cache', weight_name))
    return model

#%%
model = read_model()

sample = pd.read_csv('../data/sample_submission.csv')

if not os.path.isdir('../test_mask'):
    os.mkdir('../test_mask')
data_path = '../data'
num_channels = 16
num_mask_channels = 10
threshold = 0.3

three_band_path = os.path.join(data_path, 'three_band')

train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))

test_ids = shapes.loc[~shapes['image_id'].isin(train_wkt['ImageId'].unique()), 'image_id']

result = []
classes = ['Buildings','Misc','roads', 'track', 'trees', 'crops', 'waterways', 'standing_water', 'large_vehicle', 'small_vehicle']
color = {1:(255,153,0), 2:(255,0,0), 3:(153,102,0), 4:(255,255,0), 5:(0,51,0), 6:(153,255,102), 
         7:(51,204,255), 8:(51,204,255),9:(255,0,255), 10:(255,0,255)}
#%%
def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

#%%
def mask2poly(predicted_mask, threashold, x_scaler, y_scaler):
    polygons = extra_functions.mask2polygons_layer(predicted_mask[0] > threashold, epsilon=0, min_area=5)

    polygons = shapely.affinity.scale(polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler, origin=(0, 0, 0))
    return shapely.wkt.dumps(polygons.buffer(2.6e-5))

#%%
for image_id in tqdm(test_ids):
    image = extra_functions.read_image_16(image_id)
    
    file_name = '{}.tif'.format(image_id)
    image_3 = tiff.imread(os.path.join(three_band_path, file_name))
    image_3 = np.transpose(image_3, (1,2,0))
    image_3 = image_3/2047*255
    image_3 = np.array(image_3, dtype=np.uint8)
    H = image.shape[1]
    W = image.shape[2]

    x_max, y_min = extra_functions._get_xmax_ymin(image_id)

    predicted_mask = extra_functions.make_prediction_cropped(model, image, initial_size=(128, 128),
                                                             final_size=(128-32, 128-32),
                                                             num_masks=num_mask_channels, num_channels=num_channels)
    
    mask_to_draw = np.zeros((H,W,3),np.uint8)
    for i in range(num_mask_channels):
        mask_to_draw[predicted_mask[i]>=threshold] = color[i+1]
        #mask_to_draw[predicted_mask[i]<threshold] = (255,188,64)
        
    image_mask = cv2.addWeighted(image_3, 0.8, mask_to_draw, 0.2, 0)
    imwrite('../test_mask/{}_mask.png'.format(image_id), mask_to_draw)
    imwrite('../test_mask/{}image.png'.format(image_id), image_3)
    imwrite('../test_mask/{}_mask_image.png'.format(image_id), image_mask)
