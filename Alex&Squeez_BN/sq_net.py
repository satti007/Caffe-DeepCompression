import os
import math
import glob
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.framework.graph_util import convert_variables_to_constants

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
file=open('logs_sq.txt','w')
def conv2d(input_layer,filters,ksize,stride,padding,scope,is_training):
	with tf.variable_scope(scope) as scope:
		return tf.contrib.layers.convolution2d(inputs=input_layer,num_outputs=filters,
	                                              kernel_size=ksize,
	                                              stride=stride,padding=padding,
	                                              normalizer_fn=tf.contrib.layers.batch_norm,
	                                              normalizer_params={'is_training': is_training},scope=scope)

def max_pool2d(input_layer,ksize,stride):
		return tf.contrib.layers.max_pool2d(inputs=input_layer,kernel_size=ksize,stride=stride,padding='VALID')

def avg_pool2d(input_layer,ksize,stride):
		return tf.contrib.layers.avg_pool2d(inputs=input_layer,kernel_size=ksize,stride=stride,padding='VALID')

def fire(input_layer,s1x1, e1x1, e3x3, bypass,scope_f,is_training):
	scope1 = scope_f + '_s1x1'
	with tf.variable_scope(scope1) as scope:
	    fire_s1x1 = tf.contrib.layers.convolution2d(inputs=input_layer,num_outputs=int(s1x1),
	                                              	kernel_size=[1,1],
	                                              	stride=[1,1],padding='VALID',
	                                              	normalizer_fn=tf.contrib.layers.batch_norm,
	                                              	normalizer_params={'is_training': is_training},scope=scope)
	scope2 = scope_f + '_e1x1'
	with tf.variable_scope(scope2) as scope:
	    fire_e1x1 = tf.contrib.layers.convolution2d(inputs=fire_s1x1,num_outputs=e1x1,
	                                              	kernel_size=[1,1],
	                                              	stride=[1,1],padding='VALID',activation_fn=None,
	                                              	normalizer_fn=tf.contrib.layers.batch_norm,
	                                              	normalizer_params={'is_training': is_training},scope=scope)
	scope3 = scope_f + '_e3x3'
	with tf.variable_scope(scope3) as scope:
	    fire_e3x3 = tf.contrib.layers.convolution2d(inputs=fire_e1x1,num_outputs=e3x3,
	                                              	kernel_size=[3,3],
	                                              	stride=[1,1],padding='SAME',activation_fn=None,
	                                              	normalizer_fn=tf.contrib.layers.batch_norm,
	                                              	normalizer_params={'is_training': is_training},scope=scope)
	output=tf.concat([fire_e1x1,fire_e3x3], axis=3)
	
	if bypass == True:
		output = tf.nn.relu(tf.add(output,input_layer))
	else:
		output = tf.nn.relu(output)
	
	return output

sq_ratio=0.125
def model(x,is_training):
	with tf.contrib.slim.arg_scope([tf.contrib.slim.model_variable, tf.contrib.slim.variable]):
		
		# TODO: Convlayer: input = 227,227,3  output=111,111,96
		layer_1 = conv2d(x,96,[7,7],[2,2],'VALID','conv1',is_training)
		
		# TODO: Convlayer: input = 111,111,96  output=111,111,128
		layer_2 = fire(layer_1, sq_ratio*128, 64, 64,False,'fire1',is_training)
		
		# TODO: poollayer: input = 111,111,128  output=55,55,128
		layer_3 = max_pool2d(layer_2,[3,3],[2,2])        
		
		# TODO: Convlayer: input = 55,55,128  output=55,55,128
		layer_4 = fire(layer_3, sq_ratio*128, 64, 64, True,'fire2',is_training)
		
		# TODO: poollayer: input = 55,55,128  output=27,27,128
		layer_5 = max_pool2d(layer_4,[3,3],[2,2])
		
		# TODO: Convlayer: input = 27,27,128  output=27,27,256
		layer_6 = fire(layer_5, sq_ratio*256, 128,128, False,'fire3',is_training)
		
		# TODO: poollayer: input = 27,27,256  output=13,13,256
		layer_7 = max_pool2d(layer_6,[3,3],[2,2])
		
		# TODO: Convlayer: input = 13,13,256  output=13,13,256
		layer_8 = fire(layer_7,sq_ratio*256, 128,128,True,'fire4',is_training)
		
		# TODO: Convlayer: input = 13,13,256  output=6,6,2
		layer_9 = conv2d(layer_8,2,[3,3],[2,2],'VALID','conv2',is_training)
		
		# TODO: poollayer: input = 6,6,2  output=1,1,2 # layer_10
		logits = avg_pool2d(layer_9,[6,6],[1,1])
		y = tf.nn.softmax(logits,name='output_node')
		
		return logits,y

def one_hot_key(name):
	y = np.zeros((1,2),dtype='int')[0]
	if 'cat' in name:
		y[0] = 1
	else:
		y[1] = 1
	return y

def read_images(train):
	if train:
		folder="augmented_train/*.png"
		batch=8
	else:
		folder="valid/*.jpg"
		batch=400
	
	for fname in random.sample(glob.glob(folder),batch):
		im = Image.open(fname);
		im = (im-np.mean(im)) / np.std(im)
		label = one_hot_key(fname)
		yield im,label

def unzip(b):
    xs, ys = zip(*b)
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

def save_weights(iters):
    Wts = [p.eval(session=sess) for p in tf.trainable_variables()]
    np.savez("Weights/Sq_weights_"+str(iters)+".npz", *Wts) 

def save_pb():
    minimal_graph = convert_variables_to_constants(sess, sess.graph_def,["output_node"])
    tf.train.write_graph(minimal_graph, '.','uncompressed_Sq.pb', as_text=False)
    os.system('gzip -c uncompressed_Sq.pb > uncompressed_Sq.pb.gz')

def load_weights(iters):
    f = np.load("Weights/Sq_weights_"+str(iters)+".npz")
    initial_weights = [f[p] for p in sorted(f.files,key=lambda s: int(s[4:]))]
    assign_ops = [w.assign(v) for w, v in zip(tf.trainable_variables(), initial_weights)]
    sess.run(tf.global_variables_initializer())
    sess.run(assign_ops)

def accuracy(y,y_,iters,flag):
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	if flag ==0 :
		print "step %d training accuracy %g" % (iters,sess.run(acc))
		file.write("step %d training accuracy %g\n" % (iters,sess.run(acc)))
	else:
		print "step %d validation accuracy %g" % (iters,sess.run(acc))
		file.write("step %d validation accuracy %g\n" % (iters,sess.run(acc)))

x = tf.placeholder(tf.float32, [None,227,227,3], name='input_node')
y_ = tf.placeholder("float",shape=[None,2])
is_training_mode=tf.placeholder(tf.bool,name='is_training')
logits,y = model(x,is_training_mode)
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def train(load=None,K=None):
	if(load):
		load_weights(K)
	iters = 0
	while True:
		iters += 1;
		xs,ys=unzip(read_images(True))
		sess.run(train_step, feed_dict={x:xs,y_:ys,is_training_mode:True})
		if iters % 50 == 0:
			loss=sess.run(cross_entropy,feed_dict={x:xs,y_:ys,is_training_mode:True})
			print "step %d training loss %g" % (iters,loss)
			file.write("step %d training loss %g\n" % (iters,loss))
		if iters % 500 == 0:
			# xs,ys=unzip(read_images(False))
			# pred=sess.run(y,feed_dict={x:xs,is_training_mode:False})
			# accuracy(pred.reshape(400,2),ys,iters,1)
			save_weights(iters)
			save_pb()

