import os
import numpy as np
import tensorflow as tf
from prune import prune
from tensorflow.contrib.layers import flatten
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework.graph_util import convert_variables_to_constants

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

mnist = input_data.read_data_sets('MNIST_data', one_hot=True,reshape=False)
index=[]
file=open('index.txt','r')
for line in file:
    index.append(int(line.strip('\n')))

test_images=np.delete(mnist.test.images,index,axis=0)
test_labels=np.delete(mnist.test.labels,index,axis=0)

TEST_KEEP_PROB = 1.0

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

x = tf.placeholder("float", shape=[None,28,28,1],name='input_node')
y_ = tf.placeholder("float",shape=[None,10],name='input_labels')

# TODO: Layer 1: Convolutional. Input = 28x28x1. Output = 28x28x6.
conv1_w = weight_variable([5,5,1,6]) 
conv1_b = bias_variable([6])
conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'SAME') + conv1_b 
conv1 = tf.nn.relu(conv1)

# TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

# TODO: Layer 2: Convolutional. Output = 10x10x16.
conv2_w = weight_variable([5,5,6,16])
conv2_b = bias_variable([16])
conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
conv2 = tf.nn.relu(conv2)

# TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 

# TODO: Flatten. Input = 5x5x16. Output = 400.
fc1 = flatten(pool_2)

# TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
fc1_w = weight_variable([400,120])
fc1_b = bias_variable([120])
fc1 = tf.matmul(fc1,fc1_w) + fc1_b
fc1 = tf.nn.relu(fc1)
keep_prob1 = tf.placeholder("float",name='keep_prob1')
fc1_drop = tf.nn.dropout(fc1, keep_prob1)

# TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
fc2_w = weight_variable([120,84])
fc2_b = bias_variable([84])
fc2 = tf.matmul(fc1_drop,fc2_w) + fc2_b
fc2 = tf.nn.relu(fc2)
keep_prob2 = tf.placeholder("float",name='keep_prob2')
fc2_drop = tf.nn.dropout(fc2, keep_prob2)

# TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
fc3_w = weight_variable([84,10])
fc3_b = bias_variable([10])
logits = tf.matmul(fc2_drop, fc3_w) + fc3_b
y = tf.nn.softmax(logits,name='output_node')

cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
l2_loss = tf.nn.l2_loss(tf.concat([tf.reshape(fc1_w, [-1]), tf.reshape(fc2_w, [-1]), tf.reshape(fc3_w, [-1])],0))
l2_weight_decay = 0.001 
loss = cross_entropy + l2_loss * l2_weight_decay
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def print_validation_accuracy():
    feed_dict = {x: mnist.validation.images, y_: mnist.validation.labels, keep_prob1: TEST_KEEP_PROB, keep_prob2: TEST_KEEP_PROB}
    print "validation accuracy %g" % sess.run(accuracy, feed_dict=feed_dict) 

def print_test_accuracy():
    feed_dict = {x:test_images, y_:test_labels, keep_prob1: TEST_KEEP_PROB, keep_prob2: TEST_KEEP_PROB}
    print "test accuracy %g" % sess.run(accuracy, feed_dict=feed_dict)

def save_weights(iter=None):
    Wts = [p.eval(session=sess) for p in tf.trainable_variables()]
    np.savez("Wts/Weights_CR_"+str(iter)+".npz", *Wts)
    # np.savez("Uncompressed_weights.npz", *Wts) 

def save_pb(flag,iter=None):
    minimal_graph = convert_variables_to_constants(sess, sess.graph_def,["output_node"])
    if flag:
        name='Pbs/CR_MNIST_'+str(iter+1)+'.pb'
        tf.train.write_graph(minimal_graph, '.',name, as_text=False)
        os.system('gzip -c '+name+' > '+name+'.gz')
    else:
        tf.train.write_graph(minimal_graph, '.','uncompressed_MNIST.pb', as_text=False)
        os.system('gzip -c uncompressed_MNIST.pb > uncompressed_MNIST.pb.gz')

def load_weights(iter=None):
    f = np.load("Wts/Weights_C_"+str(iter)+".npz")
    initial_weights = [f[p] for p in sorted(f.files,key=lambda s: int(s[4:]))]
    assign_ops = [w.assign(v) for w, v in zip(tf.trainable_variables(), initial_weights)]
    sess.run(tf.global_variables_initializer())
    sess.run(assign_ops)
    
def train(iterations=100000, kp1=0.5, kp2=0.5):
    for i in range(iterations):
		batch_xs, batch_ys = mnist.train.next_batch(50)    	
		if i % 100 == 0:
			feed_dict = {x: batch_xs, y_: batch_ys, keep_prob1: kp1, keep_prob2: kp2}
			train_loss = sess.run(loss, feed_dict=feed_dict)
			print "step %d training loss %g" % (i, train_loss)
		if i % 1000==0:
			print_validation_accuracy()
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob1: kp1, keep_prob2: kp2})
    print_test_accuracy()
    
if __name__ == '__main__':
    # train(10000)
    # save_weights()
    # save_pb(False)
    kp1=0.5
    kp2=0.5
    for iter in range(0,5):
        kp1,kp2=prune(iter+1,kp1,kp2)
        load_weights(iter+1)
        train(10000,kp1,kp2)
        save_weights(iter+1)
        save_pb(True,iter)