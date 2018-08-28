from __future__ import print_function
from imutils.video import VideoStream
import numpy as np
import os
import imutils
import time
import cv2
import tensorflow as tf
#%%
#root_dir = '/home/pradeepr/Desktop/GOI_challenge/Niti_Ayog/AI_for_road_5/Object_detection/tensorflow_master/models/research/deeplab/datasets/cityscapes_india'
slim = tf.contrib.slim
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_string('cityscapes_india_root',None,'Cityscapes dataset root folder.')

path_to_images = cityscapes_india_root + '/exp/train_on_train_set/vis/segmentation_results'
path_to_video =  cityscapes_india_root + '/exp/train_on_train_set/vis'
path_to_transparent_images = cityscapes_india_root + '/exp/train_on_train_set/vis/segmentation_results_transparent'

#%%
#path_to_images = '/home/neetesh/Downloads/pic'
image_names = sorted(os.listdir(path_to_images))
image_ids =[]
for name in image_names:
    image_id = name.split('_')[0]
    image_ids.append(image_id)
    
image_ids = sorted(np.unique(image_ids))

#%%
#For making video of the transparent images
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(path_to_video + '/Results_video.avi',fourcc, 10, ( 1280,720))

# for id_ in image_ids:
    # print(id_)
    # img = cv2.imread(path_to_images+'/{}_image.png'.format(id_))
    # mask = cv2.imread(path_to_images+'/{}_prediction.png'.format(id_))
    # frame = cv2.addWeighted(img, 0.6, mask,0.4, 0)
    # #cv2.imread(path_to_transparent_images + '/' + id_ + '.png',frame)

    # out.write(frame)

    # cv2.imshow('frame',frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break
   
# # Release everything if job is finished
# out.release()
# cv2.destroyAllWindows()

#%%
#Code for generating transparent images
for id_ in image_ids:
    print(id_)
    img = cv2.imread(path_to_images+'/{}_image.png'.format(id_))
    mask = cv2.imread(path_to_images+'/{}_prediction.png'.format(id_))
    frame = cv2.addWeighted(img, 0.6, mask,0.4, 0)
    cv2.imwrite(path_to_transparent_images + '/' + id_ + '.png',frame)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   

# Release everything if job is finished

out.release()
cv2.destroyAllWindows()
#%%
