#From Git Bash run these commands 

#Install following packages
https://www.atlassian.com/git/tutorials/install-git#windows
https://gitforwindows.org/

#from  IIIT_H/models/research/deeplab/datasets
#run following command

#Add below lines in checkpoint file in path/to/file/research/deeplab/datasets/cityscapes_india/exp/train_on_train_set/train/checkpoint

model_checkpoint_path:"path/to/folder/models/research/deeplab/datasets/cityscapes_india/exp/train_on_train_set/train/model.ckpt-120000"

all_model_checkpoint_paths:"path/to/folder/models/research/deeplab/datasets/cityscapes_india/exp/train_on_train_set/train/model.ckpt-120000"


#Run following command from path/to/folder/IIIT_H/models/research/deeplab/datasets
sh convert_cityscapes_india.sh

Please add following paths in PYTHONPATH as well as system path
1)  Add deeplab folder path in sys.path.append(r'C:\Users\pradeepr\Desktop\Movehack_2018\IIIT_H\deeplab')
2)  Add slim direction path to your system path like this sys.path.append(r'C:\Program Files\Anaconda3\Lib\site-packages\tensorflow\contrib\slim')
3)  Add this path path\to\folder\IIIT_H\models\research\slim also in environment variables by name of PYTHONPATH
4)  Add this path path\to\folder\IIIT_H\models\research\ also in in environment variables by name of PYTHONPATH
5)  Add this path path\to\folder\IIIT_H\models\research\deeplab\datasets\cityscapes_india in PYTHONPATH as well as System Path

#Put your training images and corresponding masks in
path\to\folder\IIIT_H\models\research\deeplab\datasets\cityscapes_india\gtFine\val\0 
path\to\folder\IIIT_H\models\research\deeplab\datasets\cityscapes_india\leftImg8bit\val\0 

#Your results will get save in
path\to\folder\IIIT_H\models\research\deeplab\datasets\cityscapes_india\exp\train_on_train_set\vis\segmentation_results



