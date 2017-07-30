import tensorflow as tf
import numpy as np
DIGITS = "456790"
LETTERS = "BFHJKLMNPRTVWX"
CHARS = LETTERS + DIGITS


##########################################
def weight(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def relu(input_layer):
	return tf.nn.relu(input_layer)



def bias(shape, name=None):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv(x, W, stride=(1, 1), padding='SAME'):
  return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],
                      padding=padding)


def max_pool(x, ksize=(3, 3), stride=(2, 2)):
  return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='SAME')


def avg_pool(x, ksize=(16, 16), stride=(1, 1)):
  return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='VALID')


#########################################

def bn(inputs, training, epsilon=1e-7):
	decay=0.999
	gamma   = tf.Variable(tf.constant(1.0, shape=[int(inputs.get_shape()[-1])]), name='gamma', trainable=True)
	beta    = tf.Variable(tf.constant(0.0, shape=[int(inputs.get_shape()[-1])]), name='beta' , trainable=True)
	pop_mean= tf.Variable(tf.constant(0.0, shape=[int(inputs.get_shape()[-1])]), name='pop_mean' , trainable=False)
	pop_var = tf.Variable(tf.constant(1.0, shape=[int(inputs.get_shape()[-1])]), name='pop_var' , trainable=False)
	if training:
		batch_mean, batch_var = tf.nn.moments(inputs, [0,1,2])
		train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1-decay))
		train_var  = tf.assign(pop_var , pop_var*decay  + batch_var*(1-decay)); #print [batch_mean, batch_var, pop_mean, pop_var]
		with tf.control_dependencies([train_mean, train_var]):
			return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon), gamma, beta, pop_mean, pop_var
	else:
		return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, epsilon), gamma, beta, pop_mean, pop_var




###### Train Mode ######
def fire(input_layer, s1x1, e1x1, e3x3, bypass = False):
  s1x1 = int(s1x1)
  #simple_layer	
  w_s1x1 = weight([1,1,int(input_layer.get_shape()[-1]),s1x1])
  #expand_layer
  w_e1x1 = weight([1,1,s1x1,e1x1])
  w_e3x3 = weight([3,3,s1x1,e3x3])
  #fire_module
  fs = conv(input_layer, w_s1x1)
  fs_bn, gamma_f1, beta_f1, pop_mean_f1, pop_var_f1 = bn(fs,True)
  fs_r = relu(fs)
  fe_1 = conv(fs_r, w_e1x1)
  fe_3 = conv(fs_r, w_e3x3)
  fe_4, gamma_f2, beta_f2, pop_mean_f2, pop_var_f2 = bn(fe_1,True)
  fe_5, gamma_f3, beta_f3, pop_mean_f3, pop_var_f3 = bn(fe_3,True)
  fe = tf.concat([fe_4,fe_5], axis=3)
  if bypass == False:
    out = relu(fe)
  else:
    out = relu(tf.add(fe, input_layer))
  return w_s1x1, w_e1x1, w_e3x3, gamma_f1, beta_f1, pop_mean_f1, pop_var_f1, gamma_f2, beta_f2, pop_mean_f2, pop_var_f2, gamma_f3, beta_f3, pop_mean_f3, pop_var_f3, out
  

def build_model():
	sq_ratio=0.125
	x = tf.placeholder(tf.float32, [None, None, None], name='input_node')
	x_expanded = tf.expand_dims(x,3)
	w1 = weight([3,3,1,64]); 
	conv1 = conv(x_expanded, w1, stride=(2,2))
	conv1_bn, gamma_1, beta_1, pop_mean_1, pop_var_1 = bn(conv1, True)
	relu1 = relu(conv1_bn)
	m_pool_1 = max_pool(relu1)
	#Fire-Stack-1
	w_s1x1_1, we1x1_1, we3x3_1, gamma_f1_1, beta_f1_1, pop_mean_f1_1, pop_var_f1_1,  gamma_f2_1, beta_f2_1, pop_mean_f2_1, pop_var_f2_1,  gamma_f3_1, beta_f3_1, pop_mean_f3_1, pop_var_f3_1, fire_1 = fire(m_pool_1, sq_ratio*128, 64, 64, False)
	w_s1x1_2, we1x1_2, we3x3_2, gamma_f1_2, beta_f1_2, pop_mean_f1_2, pop_var_f1_2,   gamma_f2_2, beta_f2_2, pop_mean_f2_2, pop_var_f2_2,   gamma_f3_2, beta_f3_2, pop_mean_f3_2, pop_var_f3_2, fire_2 = fire(fire_1,   sq_ratio*128, 64, 64, True )
	m_pool_2 = max_pool(fire_2)
	#Fire-Stack-2
	w_s1x1_3, we1x1_3, we3x3_3, gamma_f1_3, beta_f1_3, pop_mean_f1_3, pop_var_f1_3,   gamma_f2_3, beta_f2_3, pop_mean_f2_3, pop_var_f2_3,  gamma_f3_3, beta_f3_3, pop_mean_f3_3, pop_var_f3_3, fire_3 = fire(m_pool_2, sq_ratio*256, 128, 128, False)
	w_s1x1_4, we1x1_4, we3x3_4, gamma_f1_4, beta_f1_4, pop_mean_f1_4, pop_var_f1_4,   gamma_f2_4, beta_f2_4, pop_mean_f2_4, pop_var_f2_4,  gamma_f3_4, beta_f3_4, pop_mean_f3_4, pop_var_f3_4, fire_4 = fire(fire_3,   sq_ratio*256, 128, 128, True)
	m_pool_3 = max_pool(fire_4)
	#Final-Fire-Module
	w_s1x1_5, we1x1_5, we3x3_5, gamma_f1_5, beta_f1_5, pop_mean_f1_5, pop_var_f1_5, gamma_f2_5, beta_f2_5, pop_mean_f2_5, pop_var_f2_5, gamma_f3_5, beta_f3_5, pop_mean_f3_5, pop_var_f3_5, fire_5 = fire(m_pool_3, sq_ratio*512,256,256, False)
	#Convolution-Final
	w2 = weight([1,1,512,14*len(CHARS)]); 
	conv2 = conv(fire_5, w2)
	conv2_bn, gamma_2, beta_2, pop_mean_2, pop_var_2 = bn(conv2, True)
	relu2 = relu(conv2_bn)
	final_pool = avg_pool(relu2)
	out = tf.reshape(final_pool,[-1, 14*len(CHARS)], name='output_node')	
	return x,out,[w1, 
								w_s1x1_1, we1x1_1, we3x3_1, 
								w_s1x1_2, we1x1_2, we3x3_2, 
								w_s1x1_3, we1x1_3, we3x3_3, 
								w_s1x1_4, we1x1_4, we3x3_4, 
								w_s1x1_5, we1x1_5, we3x3_5, 
								w2, 
                gamma_1, beta_1, pop_mean_1, pop_var_1,
                gamma_f1_1, beta_f1_1, pop_mean_f1_1, pop_var_f1_1, 
                gamma_f2_1, beta_f2_1, pop_mean_f2_1, pop_var_f2_1, 
                gamma_f3_1, beta_f3_1, pop_mean_f3_1, pop_var_f3_1, 
                gamma_f1_2, beta_f1_2, pop_mean_f1_2, pop_var_f1_2, 
                gamma_f2_2, beta_f2_2, pop_mean_f2_2, pop_var_f2_2, 
                gamma_f3_2, beta_f3_2, pop_mean_f3_2, pop_var_f3_2,
                gamma_f1_3, beta_f1_3, pop_mean_f1_3, pop_var_f1_3, 
                gamma_f2_3, beta_f2_3, pop_mean_f2_3, pop_var_f2_3, 
                gamma_f3_3, beta_f3_3, pop_mean_f3_3, pop_var_f3_3,
                gamma_f1_4, beta_f1_4, pop_mean_f1_4, pop_var_f1_4, 
                gamma_f2_4, beta_f2_4, pop_mean_f2_4, pop_var_f2_4, 
                gamma_f3_4, beta_f3_4, pop_mean_f3_4, pop_var_f3_4,
                gamma_f1_5, beta_f1_5, pop_mean_f1_5, pop_var_f1_5, 
                gamma_f2_5, beta_f2_5, pop_mean_f2_5, pop_var_f2_5, 
                gamma_f3_5, beta_f3_5, pop_mean_f3_5, pop_var_f3_5,
                gamma_2, beta_2, pop_mean_2, pop_var_2]





###### Test Mode ######
def fire_f(input_layer, s1x1, e1x1, e3x3, bypass = False):
  s1x1 = int(s1x1)
  #simple_layer	
  w_s1x1 = weight([1,1,int(input_layer.get_shape()[-1]),s1x1])
  #expand_layer
  w_e1x1 = weight([1,1,s1x1,e1x1])
  w_e3x3 = weight([3,3,s1x1,e3x3])
  #fire_module
  fs = conv(input_layer, w_s1x1)
  fs_bn, gamma_f1, beta_f1, pop_mean_f1, pop_var_f1 = bn(fs,False)
  fs_r = relu(fs)
  fe_1 = conv(fs_r, w_e1x1)
  fe_3 = conv(fs_r, w_e3x3)
  fe_4, gamma_f2, beta_f2, pop_mean_f2, pop_var_f2 = bn(fe_1,False)
  fe_5, gamma_f3, beta_f3, pop_mean_f3, pop_var_f3 = bn(fe_3,False)
  fe = tf.concat([fe_4,fe_5], axis=3)
  if bypass == False:
    out = relu(fe)
  else:
    out = relu(tf.add(fe, input_layer))
  return w_s1x1, w_e1x1, w_e3x3, gamma_f1, beta_f1, pop_mean_f1, pop_var_f1, gamma_f2, beta_f2, pop_mean_f2, pop_var_f2, gamma_f3, beta_f3, pop_mean_f3, pop_var_f3, out
  


def build_model_f():
	sq_ratio=0.125
	x = tf.placeholder(tf.float32, [None, None, None], name='input_node')
	x_expanded = tf.expand_dims(x,3)
	w1 = weight([3,3,1,64]); 
	conv1 = conv(x_expanded, w1, stride=(2,2))
	conv1_bn, gamma_1, beta_1, pop_mean_1, pop_var_1 = bn(conv1, False)
	relu1 = relu(conv1_bn)
	m_pool_1 = max_pool(relu1)
	#Fire-Stack-1
	w_s1x1_1, we1x1_1, we3x3_1, gamma_f1_1, beta_f1_1, pop_mean_f1_1, pop_var_f1_1,  gamma_f2_1, beta_f2_1, pop_mean_f2_1, pop_var_f2_1,  gamma_f3_1, beta_f3_1, pop_mean_f3_1, pop_var_f3_1, fire_1 = fire_f(m_pool_1, sq_ratio*128, 64, 64, False)
	w_s1x1_2, we1x1_2, we3x3_2, gamma_f1_2, beta_f1_2, pop_mean_f1_2, pop_var_f1_2,   gamma_f2_2, beta_f2_2, pop_mean_f2_2, pop_var_f2_2,   gamma_f3_2, beta_f3_2, pop_mean_f3_2, pop_var_f3_2, fire_2 = fire_f(fire_1,   sq_ratio*128, 64, 64, True )
	m_pool_2 = max_pool(fire_2)
	#Fire-Stack-2
	w_s1x1_3, we1x1_3, we3x3_3, gamma_f1_3, beta_f1_3, pop_mean_f1_3, pop_var_f1_3,   gamma_f2_3, beta_f2_3, pop_mean_f2_3, pop_var_f2_3,  gamma_f3_3, beta_f3_3, pop_mean_f3_3, pop_var_f3_3, fire_3 = fire_f(m_pool_2, sq_ratio*256, 128, 128, False)
	w_s1x1_4, we1x1_4, we3x3_4, gamma_f1_4, beta_f1_4, pop_mean_f1_4, pop_var_f1_4,   gamma_f2_4, beta_f2_4, pop_mean_f2_4, pop_var_f2_4,  gamma_f3_4, beta_f3_4, pop_mean_f3_4, pop_var_f3_4, fire_4 = fire_f(fire_3,   sq_ratio*256, 128, 128, True)
	m_pool_3 = max_pool(fire_4)
	#Final-Fire-Module
	w_s1x1_5, we1x1_5, we3x3_5, gamma_f1_5, beta_f1_5, pop_mean_f1_5, pop_var_f1_5, gamma_f2_5, beta_f2_5, pop_mean_f2_5, pop_var_f2_5, gamma_f3_5, beta_f3_5, pop_mean_f3_5, pop_var_f3_5, fire_5 = fire_f(m_pool_3, sq_ratio*512,256,256, False)
	#Convolution-Final
	w2 = weight([1,1,512,14*len(CHARS)]); 
	conv2 = conv(fire_5, w2)
	conv2_bn, gamma_2, beta_2, pop_mean_2, pop_var_2 = bn(conv2, False)
	relu2 = relu(conv2_bn)
	final_pool = avg_pool(relu2)
	out = tf.reshape(final_pool,[-1, 14*len(CHARS)], name='output_node')	
	return x,out,[w1, 
								w_s1x1_1, we1x1_1, we3x3_1, 
								w_s1x1_2, we1x1_2, we3x3_2, 
								w_s1x1_3, we1x1_3, we3x3_3, 
								w_s1x1_4, we1x1_4, we3x3_4, 
								w_s1x1_5, we1x1_5, we3x3_5, 
								w2, 
                gamma_1, beta_1, pop_mean_1, pop_var_1,
                gamma_f1_1, beta_f1_1, pop_mean_f1_1, pop_var_f1_1, 
                gamma_f2_1, beta_f2_1, pop_mean_f2_1, pop_var_f2_1, 
                gamma_f3_1, beta_f3_1, pop_mean_f3_1, pop_var_f3_1, 
                gamma_f1_2, beta_f1_2, pop_mean_f1_2, pop_var_f1_2, 
                gamma_f2_2, beta_f2_2, pop_mean_f2_2, pop_var_f2_2, 
                gamma_f3_2, beta_f3_2, pop_mean_f3_2, pop_var_f3_2,
                gamma_f1_3, beta_f1_3, pop_mean_f1_3, pop_var_f1_3, 
                gamma_f2_3, beta_f2_3, pop_mean_f2_3, pop_var_f2_3, 
                gamma_f3_3, beta_f3_3, pop_mean_f3_3, pop_var_f3_3,
                gamma_f1_4, beta_f1_4, pop_mean_f1_4, pop_var_f1_4, 
                gamma_f2_4, beta_f2_4, pop_mean_f2_4, pop_var_f2_4, 
                gamma_f3_4, beta_f3_4, pop_mean_f3_4, pop_var_f3_4,
                gamma_f1_5, beta_f1_5, pop_mean_f1_5, pop_var_f1_5, 
                gamma_f2_5, beta_f2_5, pop_mean_f2_5, pop_var_f2_5, 
                gamma_f3_5, beta_f3_5, pop_mean_f3_5, pop_var_f3_5,
                gamma_2, beta_2, pop_mean_2, pop_var_2]

