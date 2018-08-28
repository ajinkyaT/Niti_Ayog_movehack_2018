#Run from below folder 
#From Git Bash run these commands
MINGW64 ~/Desktop/Movehack_2018/IIIT_H/models/research/deeplab/datasets

sh convert_cityscapes_india.sh

Please do following things in your window machine
1)  Add deeplab folder path in sys.path.append(r'C:\Users\pradeepr\Desktop\Movehack_2018\IIIT_H\deeplab')
2)  Add slim direction path to your system path like this sys.path.append('C:\Program Files\Anaconda3\Lib\site-packages\tensorflow\contrib\slim')
3)  Add this path C:\Users\pradeepr\Desktop\Movehack_2018\IIIT_H\models\research\slim also in environment variables by name of PYTHONPATH
4)  Add this path C:\Users\pradeepr\Desktop\Movehack_2018\IIIT_H\models\research\ also in in environment variables by name of PYTHONPATH
5)  Add this path C:\Users\pradeepr\Desktop\Movehack_2018\IIIT_H\models\research\deeplab\datasets\cityscapes_india in PYTHONPATH as well as System Path
6)  Run preparation.py C:\Users\pradeepr\Desktop\Movehack_2018\IIIT_H\models\research\deeplab\datasets\cityscapes_india\cityscapesscripts\preparation to generate masks


