import os
import math
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework.graph_util import convert_variables_to_constants

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

tf.GraphKeys.PRUNING_MASKS = "pruning_masks"

x = tf.placeholder("float", shape=[None,28,28,1],name='input_node')
y_ = tf.placeholder("float",shape=[None,10],name='input_labels')


# TODO: Layer 1: Convolutional. Input = 28x28x1. Output = 28x28x6.
conv1_w = weight_variable([5,5,1,6]) 
# prune_conv1 = tf.Variable(tf.ones_like(conv1_w),trainable=False, collections=[tf.GraphKeys.PRUNING_MASKS],name='prune_conv1')
# conv1_pruned = tf.multiply(conv1_w, prune_conv1) #element-wise multiplication
conv1_b = bias_variable([6])
# conv1 = tf.nn.conv2d(x,conv1_pruned, strides = [1,1,1,1], padding = 'SAME') + conv1_b 
conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'SAME') + conv1_b 
conv1 = tf.nn.relu(conv1)

# TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

# TODO: Layer 2: Convolutional. Output = 10x10x16.
conv2_w = weight_variable([5,5,6,16])
# prune_conv2 = tf.Variable(tf.ones_like(conv2_w),trainable=False, collections=[tf.GraphKeys.PRUNING_MASKS],name='prune_conv2')
# conv2_pruned = tf.multiply(conv2_w, prune_conv2) #element-wise multiplication
conv2_b = bias_variable([16])
conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
conv2 = tf.nn.relu(conv2)

# TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 

# TODO: Flatten. Input = 5x5x16. Output = 400.
fc1 = flatten(pool_2)

# TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
fc1_w = weight_variable([400,120])
prune_mask1 = tf.Variable(tf.ones_like(fc1_w),trainable=False, collections=[tf.GraphKeys.PRUNING_MASKS],name='prune_mask1')
fc1_pruned = tf.multiply(fc1_w, prune_mask1) #element-wise multiplication
fc1_b = bias_variable([120])
fc1 = tf.matmul(fc1,fc1_pruned) + fc1_b
# fc1 = tf.matmul(fc1,fc1_w) + fc1_b
fc1 = tf.nn.relu(fc1)
keep_prob1 = tf.placeholder("float",name='keep_prob1')
fc1_drop = tf.nn.dropout(fc1, keep_prob1)

# TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
fc2_w = weight_variable([120,84])
prune_mask2 = tf.Variable(tf.ones_like(fc2_w), trainable=False, collections=[tf.GraphKeys.PRUNING_MASKS],name='prune_mask2')
fc2_pruned = tf.multiply(fc2_w, prune_mask2)
fc2_b = bias_variable([84])
fc2 = tf.matmul(fc1_drop,fc2_pruned) + fc2_b
# fc2 = tf.matmul(fc1_drop,fc2_w) + fc2_b
fc2 = tf.nn.relu(fc2)
keep_prob2 = tf.placeholder("float",name='keep_prob2')
fc2_drop = tf.nn.dropout(fc2, keep_prob2)

# TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
fc3_w = weight_variable([84,10])
prune_mask3 = tf.Variable(tf.ones_like(fc3_w), trainable=False, collections=[tf.GraphKeys.PRUNING_MASKS],name='prune_mask3')
fc3_pruned = tf.multiply(fc3_w, prune_mask3)
fc3_b = bias_variable([10])
logits = tf.matmul(fc2_drop,fc3_pruned) + fc3_b
# logits = tf.matmul(fc2_drop,fc3_w) + fc3_b
y = tf.nn.softmax(logits, name='output_node')

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_)
# l2_loss = tf.nn.l2_loss(tf.concat([tf.reshape(conv1_w, [-1]),tf.reshape(conv2_w, [-1]),tf.reshape(fc1_w, [-1]), tf.reshape(fc2_w, [-1]), tf.reshape(fc3_w, [-1])],0))
l2_loss = tf.nn.l2_loss(tf.concat([tf.reshape(fc1_w, [-1]), tf.reshape(fc2_w, [-1]), tf.reshape(fc3_w, [-1])],0))
l2_weight_decay = 0.001 
loss = cross_entropy + l2_loss * l2_weight_decay
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

threshold = 0.0001
t1 = tf.sqrt(tf.nn.l2_loss(fc1_w)) * threshold
t2 = tf.sqrt(tf.nn.l2_loss(fc2_w)) * threshold
t3 = tf.sqrt(tf.nn.l2_loss(fc3_w)) * threshold
# t4 = tf.sqrt(tf.nn.l2_loss(conv1_w)) * threshold
# t5 = tf.sqrt(tf.nn.l2_loss(conv2_w)) * threshold

nonzero_indicator1 = tf.to_float(tf.not_equal(fc1_w, tf.zeros_like(fc1_w)))
nonzero_indicator2 = tf.to_float(tf.not_equal(fc2_w, tf.zeros_like(fc2_w)))
nonzero_indicator3 = tf.to_float(tf.not_equal(fc3_w, tf.zeros_like(fc3_w)))
# nonzero_indicator4 = tf.to_float(tf.not_equal(conv1_w, tf.zeros_like(conv1_w)))
# nonzero_indicator5 = tf.to_float(tf.not_equal(conv2_w, tf.zeros_like(conv2_w)))
count_parameters1 = tf.reduce_sum(nonzero_indicator1)
count_parameters2 = tf.reduce_sum(nonzero_indicator2)
count_parameters3 = tf.reduce_sum(nonzero_indicator3)
# count_parameters4 = tf.reduce_sum(nonzero_indicator4)
# count_parameters5 = tf.reduce_sum(nonzero_indicator5)



indicator_matrix1 = tf.multiply(tf.to_float(tf.greater_equal(fc1_w, tf.ones_like(fc1_w) * t1)), prune_mask1)
indicator_matrix2 = tf.multiply(tf.to_float(tf.greater_equal(fc2_w, tf.ones_like(fc2_w) * t2)), prune_mask2)
indicator_matrix3 = tf.multiply(tf.to_float(tf.greater_equal(fc3_w, tf.ones_like(fc3_w) * t3)), prune_mask3)
# indicator_matrix4 = tf.multiply(tf.to_float(tf.greater_equal(conv1_w, tf.ones_like(conv1_w) * t4)), prune_conv1)
# indicator_matrix5 = tf.multiply(tf.to_float(tf.greater_equal(conv2_w, tf.ones_like(conv2_w) * t5)), prune_conv2)

update_mask1 = tf.assign(prune_mask1, indicator_matrix1)
update_mask2 = tf.assign(prune_mask2, indicator_matrix2)
update_mask3 = tf.assign(prune_mask3, indicator_matrix3)
# update_mask4 = tf.assign(prune_conv1, indicator_matrix4)
# update_mask5 = tf.assign(prune_conv2, indicator_matrix5)
# update_all_masks = tf.group(update_mask1, update_mask2, update_mask3, update_mask4, update_mask5)
update_all_masks = tf.group(update_mask1, update_mask2, update_mask3)

prune_fc1 = fc1_w.assign(fc1_pruned)
prune_fc2 = fc2_w.assign(fc2_pruned)
prune_fc3 = fc3_w.assign(fc3_pruned)
# prune_conv1w = conv1_w.assign(conv1_pruned)
# prune_conv2w = conv2_w.assign(conv2_pruned)
# prune_all = tf.group(prune_fc1, prune_fc2, prune_fc3,prune_conv1w,prune_conv2w)
prune_all = tf.group(prune_fc1, prune_fc2, prune_fc3)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.PRUNING_MASKS)))

def print_validation_accuracy(iter):
    feed_dict = {x: mnist.validation.images, y_: mnist.validation.labels, keep_prob1: TEST_KEEP_PROB, keep_prob2: TEST_KEEP_PROB}
    print("step %d validation accuracy %g" % (iter,sess.run(accuracy, feed_dict=feed_dict)))

def print_test_accuracy():
    feed_dict = {x:test_images, y_:test_labels, keep_prob1: TEST_KEEP_PROB, keep_prob2: TEST_KEEP_PROB}
    print("test accuracy %g" % sess.run(accuracy, feed_dict=feed_dict))

def train(iterations=10000, kp1=0.5, kp2=0.5):
    print('######### Started Training ##########')
    for i in range(iterations):
        batch_xs, batch_ys = mnist.train.next_batch(50)     
        if i % 100 == 0:
            feed_dict = {x: batch_xs, y_: batch_ys, keep_prob1: TEST_KEEP_PROB, keep_prob2: TEST_KEEP_PROB}
            train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            print("step %d training accuracy %g" % (i, train_accuracy))
        if i % 1000==0:
            print_validation_accuracy(i)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob1: kp1, keep_prob2: kp2})
    print_test_accuracy()
    print('######### Done Training ##########')

def print_parameter_counts():
    print("W Parameter Counts:")
    print("FC1_Parameters: {0}".format(sess.run(count_parameters1)))
    print("FC2_Parameters: {0}".format(sess.run(count_parameters2)))
    print("FC3_Parameters: {0}".format(sess.run(count_parameters3)))

def calculate_new_keep_prob(original_keep_prob, original_connections, retraining_connections):
    return 1.0 - ((1.0 - original_keep_prob) * math.sqrt(retraining_connections / original_connections))

def compress(times=1):
    kp1 = kp2 = 0.5
    for i in range(times):
        print('######### Start ##########')
        print("Compressing iteration " +str(i + 1))
        c1 = sess.run(count_parameters1)
        c2 = sess.run(count_parameters2)
        print("Before pruning")
        print_parameter_counts()
        sess.run(update_all_masks)
        sess.run(prune_all)
        c1_retrain = sess.run(count_parameters1)
        c2_retrain = sess.run(count_parameters2)
        kp1 = calculate_new_keep_prob(kp1, c1, c1_retrain)
        kp2 = calculate_new_keep_prob(kp2, c2, c2_retrain)
        if(i==4):
            continue
        else:
            train(10000, kp1, kp2)
        print("After pruning")
        print_parameter_counts()
        print('######### Done ##########')
        # save_wts(i)
        write_to_pb(i)

def save_wts(k):
    w = [p.eval(sess) for p in tf.trainable_variables()]
    np.savez('./Wts_{}.npz'.format(k), *w)

def write_to_pb(iter):
    # name='models2/compressed_conv_MNIST_'+str(iter+1)+'.pb'
    minimal_graph = convert_variables_to_constants(sess,sess.graph_def,["output_node"])
    tf.train.write_graph(minimal_graph,'.','z.pb', as_text=False)
    os.system('gzip -c '+name+' > '+name+'.gz')

if __name__ == '__main__':
    sess.run(tf.global_variables_initializer())
    train()
    compress(5)
