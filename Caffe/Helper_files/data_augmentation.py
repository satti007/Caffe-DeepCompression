import os
import cv2
import glob
import numpy as np

os.system('rm -rf ../input/augmented_train')
os.system('mkdir ../input/augmented_train')

def get_patches(image,img_path,idx,flag):
	animal=img_path.split("/")[-1].split(".")[0]

	# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
	crop_img=image[0:227,0:227]#top-left
	cv2.imwrite("../input/augmented_train/"+animal+"_"+str(idx)+"_"+str(flag)+"_1.png",crop_img)
	
	crop_img=image[31:258,0:227]#bottom-left
	cv2.imwrite("../input/augmented_train/"+animal+"_"+str(idx)+"_"+str(flag)+"_2.png",crop_img)

	crop_img=image[0:227,31:258]#top-right
	cv2.imwrite("../input/augmented_train/"+animal+"_"+str(idx)+"_"+str(flag)+"_3.png",crop_img)

	crop_img=image[31:258,31:258]#bottom-right
	cv2.imwrite("../input/augmented_train/"+animal+"_"+str(idx)+"_"+str(flag)+"_4.png",crop_img)

	crop_img=image[16:243,16:243]#center
	cv2.imwrite("../input/augmented_train/"+animal+"_"+str(idx)+"_"+str(flag)+"_5.png",crop_img)
	
	print("Done ",flag,img_path)

data = sorted([img for img in glob.glob("../input/train_valid/*")])

resize_dim=(258,258)

for in_idx, img_path in enumerate(data):
	if in_idx%11==0:
		continue
	image = cv2.imread(img_path)
	resized = cv2.resize(image,resize_dim)
	mirror_img= cv2.flip(resized,1)
	get_patches(resized,img_path,in_idx,0)
	get_patches(mirror_img,img_path,in_idx,1)

