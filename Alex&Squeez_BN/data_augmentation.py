import os
import cv2
import glob
import numpy as np

os.system('rm -rf augmented_train')
os.system('mkdir augmented_train')

data = sorted([img for img in glob.glob("train/*")])

for in_idx, img_path in enumerate(data):
	animal=img_path.split("/")[-1].split(".")[0]
	image = cv2.imread(img_path)
	cv2.imwrite("augmented_train/"+animal+"_"+str(in_idx)+"_1.png",image)
	mirror_img= cv2.flip(image,1)
	cv2.imwrite("augmented_train/"+animal+"_"+str(in_idx)+"_2.png",mirror_img)
	print 'Done:',img_path,in_idx	
