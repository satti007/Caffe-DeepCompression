import os
import glob
import caffe
import numpy as np


print 'Started Predictions.......'

caffe_root = '/home/satish_kumar/caffe-master/'  #path to caffe root directory
caffe.set_mode_cpu() #setting the mode as cpu(if you have installed caffe with gpu support,use gpu) 
model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


# Preprocessing
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})# create transformer for the input called 'data'
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
net.blobs['data'].reshape(1,         # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227


test_data = [img for img in glob.glob("../input/test/*jpg")] #path to testing images
file=open('caffenet_predictions.txt','w')  #file to write the predictions
for idx,img_path in enumerate(test_data):
	image = caffe.io.load_image(img_path)  # load the image 
	transformed_image = transformer.preprocess('data',image) # preprocess according to above steps
	net.blobs['data'].data[...] = transformed_image# copy the image data into the memory allocated for the net
	# perform classification
	output = net.forward()
	output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
	# print 'predicted class is:', output_prob.argmax()
	labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
	labels = np.loadtxt(labels_file, str, delimiter='\t')
	print labels[output_prob.argmax()]
	file.write("%s\n"%labels[output_prob.argmax()])

print 'Done predictions.....'
