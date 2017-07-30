import os
import math
import glob
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.framework.graph_util import convert_variables_to_constants

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
file=open('logs_alex.txt','w')

def conv2d(input_layer,filters,ksize,stride,padding,scope,is_training):
  with tf.variable_scope(scope) as scope:
    return tf.contrib.layers.convolution2d(inputs=input_layer,num_outputs=filters,
                                                kernel_size=ksize,
                                                stride=stride,padding=padding,
                                                normalizer_fn=tf.contrib.layers.batch_norm,
                                                normalizer_params={'is_training': is_training},scope=scope)

def max_pool2d(input_layer,ksize,stride):
    return tf.contrib.layers.max_pool2d(inputs=input_layer,kernel_size=ksize,stride=stride,padding='VALID')

def FC(input_layer,neurons,scope):
  with tf.variable_scope(scope) as scope:
    return tf.contrib.layers.fully_connected(inputs=input_layer,num_outputs=neurons,scope=scope)

def model(x,is_training):
  with tf.contrib.slim.arg_scope([tf.contrib.slim.model_variable, tf.contrib.slim.variable]):
    
    # TODO: Convlayer: input = 227,227,3  output=55,55,96
    layer_1 = conv2d(x,96,[11,11],[4,4],'VALID','conv1',is_training)
    
    # TODO: poollayer: input = 55,55,96  output=27,27,96
    pool_1 = max_pool2d(layer_1,[3,3],[2,2])
    
    # TODO: Convlayer: input = 27,27,96 output=27,27,256
    layer_2 = conv2d(pool_1,256,[5,5],[1,1],'SAME','conv2',is_training)        
    
    # TODO: poollayer: input = 27,27,256  output=13,13,256
    pool_2 = max_pool2d(layer_2,[3,3],[2,2])
    
    # TODO: Convlayer: input = 13,13,256  output=13,13,384
    layer_3 = conv2d(pool_2,384,[3,3],[1,1],'SAME','conv3',is_training)
    
    # TODO: Convlayer: input = 13,13,384  output=13,13,384
    layer_4 = conv2d(layer_3,384,[3,3],[1,1],'SAME','conv4',is_training)
    
    # TODO: Convlayer: input = 13,13,384  output=13,13,256
    layer_5 = conv2d(layer_4,256,[3,3],[1,1],'SAME','conv5',is_training)
    
    # TODO: poollayer: input = 13,13,256  output=6,6,256
    pool_3 = max_pool2d(layer_5,[3,3],[2,2])
    
    layer_f = tf.contrib.layers.flatten(pool_3)
    
    # TODO: FC layer: input = 9216 output=4096
    layer_6 = FC(layer_f,4096,'fc6')
    
    # TODO: FC layer: input = 4096  output=4096
    layer_7 = FC(layer_6,4096,'fc7')
    
    # TODO: FC layer: input = 4096  output=2
    logits = FC(layer_7,2,'fc8')
    
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
    folder="net_test/*.jpg"
    batch=8
  else:
    folder="valid/*.jpg"
    batch=500
  
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
    np.savez("Weights1/Alex_weights_"+str(iters)+".npz", *Wts) 

def save_pb():
    minimal_graph = convert_variables_to_constants(sess, sess.graph_def,["output_node"])
    tf.train.write_graph(minimal_graph, '.','uncompressed_Alex.pb', as_text=False)
    os.system('gzip -c uncompressed_Alex.pb > uncompressed_Alex.pb.gz')

def load_weights(iters):
    f = np.load("Weights1/Alex_weights_"+str(iters)+".npz")
    initial_weights = [f[p] for p in sorted(f.files,key=lambda s: int(s[4:]))]
    assign_ops = [w.assign(v) for w, v in zip(tf.trainable_variables(), initial_weights)]
    sess.run(tf.global_variables_initializer())
    sess.run(assign_ops)

def accuracy(y,y_,iters):
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
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

for i in tf.trainable_variables():
  print i

def train(load=None,K=None):
  if(load):
    load_weights(K)
  try:
    iters = 0
    while True:
      iters += 1;
      xs,ys=unzip(read_images(True))
      sess.run(train_step, feed_dict={x:xs,y_:ys,is_training_mode:True})
      if iters % 5 == 0:
        loss=sess.run(cross_entropy,feed_dict={x:xs,y_:ys,is_training_mode:True})
        print "step %d training loss %g" % (iters,loss)
        file.write("step %d training loss %g\n" % (iters,loss))
      
      # if iters % 500 == 0:
      #   xs,ys=unzip(read_images(False))
      #   pred=sess.run(y,feed_dict={x:xs,is_training_mode:False})
      #   accuracy(pred.reshape(500,2),ys,iters)
      #   save_weights(iters)
      #   save_pb()
  except KeyboardInterrupt: 
    # save_weights(iters)
    save_pb()

train()

