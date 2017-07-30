import os
import cv2
import glob
import random
import numpy as np
from PIL import Image

data = [img for img in glob.glob("../input/images/*")]

os.system('rm -rf train/')
os.system('rm -rf valid/')
os.system('rm -rf test/')
os.system('mkdir train')
os.system('mkdir valid')
os.system('mkdir test')

test = os.listdir('../input/test')

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    return img

count=0
for in_idx, img_path in enumerate(data):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    img = Image.fromarray(img)
    name = img_path.split('/')[1]
    if name in test:
        img.save('test/'+name)
    else:
    	count=count+1
    	if count % 11 == 0:
    		img.save('valid/'+name)
    	else:
    		img.save('train/'+name)
    print 'Finished '+img_path

print 'Finished processing all images'
print 'train_cats: ' ,len(glob.glob1('train', "cat*"))
print 'train_dogs: ' ,len(glob.glob1('train', "dog*"))
print 'valid_cats: ' ,len(glob.glob1('valid', "cat*"))
print 'valid_dogs: ' ,len(glob.glob1('valid', "dog*"))
print 'test_cats: '  ,len(glob.glob1('test' , "cat*"))
print 'test_dogs: '  ,len(glob.glob1('test' , "dog*"))
