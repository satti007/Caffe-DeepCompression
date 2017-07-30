import os
import glob
import random
import numpy as np

indicies=random.sample(range(0,12500),2500)
cat_test_indicies,null,null=np.split(indicies,[1500,1500])
dog_test_indicies=cat_test_indicies+12500

data = sorted([img for img in glob.glob("../input/images/*")])
test_labels=[]
train_valid_labels=[]

os.system('rm -rf ../input/test')
os.system('rm -rf ../input/train_valid')
os.system('mkdir ../input/test')
os.system('mkdir ../input/train_valid')

for in_idx, img_path in enumerate(data):
	if in_idx in cat_test_indicies or in_idx in dog_test_indicies: 
		os.system('cp '+img_path+' ../input/test')
		print img_path,' added to test'
		if 'cat' in img_path:
			test_labels.append(0)
		else:
			test_labels.append(1)
	else:
		os.system('cp '+img_path+' ../input/train_valid')
		print img_path,' added to train_valid'
		if 'cat' in img_path:
			train_valid_labels.append(0)
		else:
			train_valid_labels.append(1)


print 'test_cats: ',test_labels.count(0)
print 'test_dogs: ',test_labels.count(1)
print 'train_valid_cats: ',train_valid_labels.count(0)
print 'train_valid_dogs: ',train_valid_labels.count(1)
