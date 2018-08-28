from __future__ import division

import numpy as np
from keras.models import Model
from keras.layers import Input, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose, activations
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

import gc
import os
import pandas as pd
import extra_functions
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
import h5py

from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate

from keras.optimizers import Nadam
from keras.callbacks import History
import pandas as pd
from keras.backend import binary_crossentropy

import datetime

import random
import threading

from keras.models import model_from_json

#%%
train_pretrained_model = False
training_fraction = 0.88
n_batch_per_epoch = 25
n_epoch = 100
img_rows = 128
img_cols = 128
batch_size = 64
smooth = 1e-12
data_path = '../data'
train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))
num_channels = 3
train_ids = np.load('../data/train_ids.npy')
# class names are in the order of class_ids = index + 1
classes = ['Buildings','roads', 'crops', 'waterways', 'large_vehicle', 'small_vehicle']
class_ids = [1, 3, 6, 7, 9, 10]
num_mask_channels = len(classes)
#%%
def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

#%%
def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1,-2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

#%%
def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)

#%%
def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[3] - refer.get_shape()[3]).value
        #assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[2] - refer.get_shape()[2]).value
        #assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)
        
        return (abs(ch1), abs(ch2)), (abs(cw1), abs(cw2))  

#%%
def get_unet0():
    concat_axis = 1
    inputs = Input(( num_channels,img_rows, img_cols))
    
    conv1 = Conv2D(32, (3, 3), padding="same", name="conv1_1", activation="elu", data_format="channels_first")(inputs)
    conv1 = BatchNormalization(axis= concat_axis)(conv1)
    conv1 = Conv2D(32, (3, 3), padding="same", activation="elu", data_format="channels_first")(conv1)
    conv1 = BatchNormalization(axis= concat_axis)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(conv1)
    
    conv2 = Conv2D(64, (3, 3), padding="same", activation="elu", data_format="channels_first")(pool1)
    conv2 = BatchNormalization(axis= concat_axis)(conv2)
    conv2 = Conv2D(64, (3, 3), padding="same", activation="elu", data_format="channels_first")(conv2)
    conv2 = BatchNormalization(axis= concat_axis)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(conv2)
    
    conv3 = Conv2D(128, (3, 3), padding="same", activation="elu", data_format="channels_first")(pool2)
    conv3 = BatchNormalization(axis= concat_axis)(conv3)
    conv3 = Conv2D(128, (3, 3), padding="same", activation="elu", data_format="channels_first")(conv3)
    conv3 = BatchNormalization(axis= concat_axis)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(conv3)

    conv4 = Conv2D(256, (3, 3), padding="same", activation="elu", data_format="channels_first")(pool3)
    conv4 = BatchNormalization(axis= concat_axis)(conv4)
    conv4 = Conv2D(256, (3, 3), padding="same", activation="elu", data_format="channels_first")(conv4)
    conv4 = BatchNormalization(axis= concat_axis)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(conv4)

    conv5 = Conv2D(512, (3, 3), padding="same", activation="elu", data_format="channels_first")(pool4)
    conv5 = BatchNormalization(axis= concat_axis)(conv5)
    conv5 = Conv2D(512, (3, 3), padding="same", activation="elu", data_format="channels_first")(conv5)
    conv5 = BatchNormalization(axis= concat_axis)(conv5)

    up_conv5 = UpSampling2D(size=(2, 2), data_format="channels_first")(conv5)
    ch, cw = get_crop_shape(up_conv5,conv4)
    crop_conv4 = Cropping2D(cropping=(ch,cw), data_format="channels_first")(conv4)
    up6   = concatenate([up_conv5,crop_conv4], axis=concat_axis)
    conv6 = Conv2D(256, (3, 3), padding="same", activation="elu", data_format="channels_first")(up6)
    conv6 = BatchNormalization(axis= concat_axis)(conv6)
    conv6 = Conv2D(256, (3, 3), padding="same", activation="elu", data_format="channels_first")(conv6)
    conv6 = BatchNormalization(axis= concat_axis)(conv6)
    
    up_conv6 = UpSampling2D(size=(2, 2), data_format="channels_first")(conv6)
    ch, cw = get_crop_shape(up_conv6,conv3)
    crop_conv3 = Cropping2D(cropping=(ch,cw), data_format="channels_first")(conv3)
    up7   = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = Conv2D(128, (3, 3), padding="same", activation="elu", data_format="channels_first")(up7)
    conv7 = BatchNormalization(axis= concat_axis)(conv7)
    conv7 = Conv2D(128, (3, 3), padding="same", activation="elu", data_format="channels_first")(conv7)
    conv7 = BatchNormalization(axis= concat_axis)(conv7)

    up_conv7 = UpSampling2D(size=(2, 2), data_format="channels_first")(conv7)
    ch, cw = get_crop_shape(up_conv7,conv2)
    crop_conv2 = Cropping2D(cropping=(ch,cw), data_format="channels_first")(conv2)
    up8   = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = Conv2D(64, (3, 3), padding="same", activation="elu", data_format="channels_first")(up8)
    conv8 = BatchNormalization(axis= concat_axis)(conv8)
    conv8 = Conv2D(64, (3, 3), padding="same", activation="elu", data_format="channels_first")(conv8)
    conv8 = BatchNormalization(axis= concat_axis)(conv8)

    up_conv8 = UpSampling2D(size=(2, 2), data_format="channels_first")(conv8)
    ch, cw = get_crop_shape(up_conv8,conv1)
    crop_conv1 = Cropping2D(cropping=(ch,cw), data_format="channels_first")(conv1)
    up9   = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = Conv2D(32, (3, 3), padding="same", activation="elu", data_format="channels_first")(up9)
    conv9 = BatchNormalization(axis= concat_axis)(conv9)
    conv9 = Conv2D(32, (3, 3), padding="same", activation="elu", data_format="channels_first")(conv9)
    conv9 = BatchNormalization(axis= concat_axis)(conv9)
    
    outputs = Conv2D(num_mask_channels, (1, 1), activation='sigmoid',data_format="channels_first")(conv9)

    model = Model(input=[inputs], output=[outputs])
    return model

#%%
def cache_train_16():
    
    print('num_train_images =', train_wkt['ImageId'].nunique())
    train_shapes = shapes[shapes['image_id'].isin(train_wkt['ImageId'].unique())]
    np.save('../data/train_ids.npy',train_shapes['image_id'])
    min_train_height = train_shapes['height'].min()
    min_train_width = train_shapes['width'].min()

    ids = []
    i = 0

    for image_id in tqdm(sorted(train_wkt['ImageId'].unique())):
        image = extra_functions.read_image_16(image_id)
        _, height, width = image.shape

        img = image[:, :min_train_height, :min_train_width]
        img_mask = extra_functions.generate_mask(image_id,
                                                     height,
                                                     width,
                                                     num_mask_channels=num_mask_channels,
                                                     train=train_wkt)[ :,:min_train_height, :min_train_width]
        

        np.save('../data/data_files/{}_img.npy'.format(image_id),img)
        np.save('../data/data_files/{}_mask.npy'.format(image_id),img_mask)                                           
        ids += [image_id]
        i += 1 
#%%
def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::1, ...]
    x = x.swapaxes(0, axis)
    return x

#%%
def generate_batch(temp_id,batch_size, horizontal_flip=False, vertical_flip=False):
    
    #Data Generator
    #temp_id = np.random.choice(train_ids,size=1,replace=True)[0]
    
    X = np.load('../data/data_files/'+temp_id+'_img.npy')
    y_temp = np.load('../data/data_files/'+temp_id+'_mask.npy')
    y = []
    for k,id_ in enumerate(class_ids):
        y.append(y_temp[id_ - 1])
    y = np.array(y)    
    X = X[13:,:,:] 
    
#    #Converting data and masks from channel first to channel last format
#    X = np.transpose(X,(1,0,2))
#    X = np.transpose(X,(0,2,1)) 
#    y = np.transpose(y,(1,0,2))
#    y = np.transpose(y,(0,2,1))
    
    X_batch = np.zeros((batch_size, num_channels,img_rows, img_cols))
    y_batch = np.zeros((batch_size,num_mask_channels, img_rows, img_cols))
    X_height = X.shape[1]
    X_width = X.shape[2]
    
    for i in range(batch_size):
        random_width = random.randint(0, X_width - img_cols - 1)
        random_height = random.randint(0, X_height - img_rows - 1)

        y_batch[i] = y[:,random_height: random_height + img_rows, random_width: random_width + img_cols]
        X_batch[i] = X[:,random_height: random_height + img_rows, random_width: random_width + img_cols]
        
    for i in range(batch_size):
        xb = X_batch[i]
        yb = y_batch[i]

        if horizontal_flip:
            if np.random.random() < 0.5:
                xb = flip_axis(xb, 1)
                yb = flip_axis(yb, 1)

        if vertical_flip:
            if np.random.random() < 0.5:
                xb = flip_axis(xb, 2)
                yb = flip_axis(yb, 2)

        X_batch[i] = xb
        y_batch[i] = yb

    del(X,y)        
    return X_batch,y_batch    #[:, :, 16:16 + img_rows - 32, 16:16 + img_cols - 32]

#%%
def save_model(model, cross=''):
    json_string = model.to_json()
    if not os.path.isdir('../cache'):
        os.mkdir('../cache')
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    open(os.path.join('../cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('../cache', weight_name), overwrite=True)

#%%
def update_and_save_history(result,suffix=''):
    
    filename = '../history/history_' + suffix + '.csv'
    
    Final_history['val_loss'].append(result.history['val_loss'][0])
    Final_history['val_binary_crossentropy'].append(result.history['val_binary_crossentropy'][0])
    Final_history['val_jaccard_coef_int'].append(result.history['val_jaccard_coef_int'][0])
    Final_history['val_acc'].append(result.history['val_acc'][0])
    Final_history['loss'].append(result.history['loss'][0])
    Final_history['binary_crossentropy'].append(result.history['binary_crossentropy'][0])
    Final_history['jaccard_coef_int'].append(result.history['jaccard_coef_int'][0])
    Final_history['acc'].append(result.history['acc'][0])
    
    pd.DataFrame(Final_history).to_csv(filename, index=False)   
#%%
def save_history(history, suffix):
    filename = '../history/history_' + suffix + '.csv'
    pd.DataFrame(history.history).to_csv(filename, index=False)

#%%
def read_model(cross=''):
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    model = model_from_json(open(os.path.join('../cache', json_name)).read())
    model.load_weights(os.path.join('../cache', weight_name))
    return model

#%%
Final_history ={'val_loss': [],
   'val_binary_crossentropy': [],
   'val_jaccard_coef_int': [],
   'val_acc': [],
   'loss': [],
   'binary_crossentropy': [],
   'jaccard_coef_int': [],
   'acc': []}
#%%
if __name__ == '__main__':
    data_path = '../data'
    now = datetime.datetime.now()
    #cache_train_16()
    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    print('[{}] Reading train...'.format(str(datetime.datetime.now())))
    history = History()
    callbacks = [ history,]
    suffix = '_rgb_less_class_'

    #temp_valid_ids, temp_train_ids = train_test_split(train_ids, test_size= int(training_fraction*len(train_ids)), random_state=100) 
    temp_train_ids = train_ids; temp_valid_ids = train_ids
    if train_pretrained_model:
        model = read_model()
        model.compile(optimizer='Adadelta', loss=jaccard_coef_loss, metrics=['binary_crossentropy', jaccard_coef_int,'accuracy'])
    else:
        #Building model
        model = get_unet0()
        model.compile(optimizer='Adadelta', loss=jaccard_coef_loss, metrics=['binary_crossentropy', jaccard_coef_int,'accuracy'])
        model.summary()    
    
    with tf.device("/device:GPU:0"): 
        
        for epoch in range(n_epoch):
            for batch in range(n_batch_per_epoch):
                print("Epoch no :"+ str(epoch+1) + " batch "+ str(batch+1))
                temp_id = train_ids[batch % n_batch_per_epoch]
                
                x_train, y_train = generate_batch(temp_id,batch_size, horizontal_flip=True, vertical_flip=True);
                x_valid,y_valid = generate_batch(temp_id,int(batch_size/4), horizontal_flip=True, vertical_flip=True);
                
#                if (epoch != 0 or batch != 0) :
#                    model = read_model()
#                    model.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['binary_crossentropy', jaccard_coef_loss,'accuracy'])
#                    
                #training the model
                history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),epochs=1,batch_size = batch_size,
                                    verbose=1, shuffle=True) 
                update_and_save_history(history)
                #save_model(model)   
                if batch%10 == 0:
                    save_model(model,"{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=n_epoch, suffix=suffix))   
                    print("saving the model.")
                gc.collect();

#%%