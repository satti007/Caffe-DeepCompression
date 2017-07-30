import os
import math
import glob
import random
import numpy as np
import sklearn.metrics
import tensorflow as tf
from PIL import Image
from sklearn.utils import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None)
    return graph

def one_hot_key(name):
    y = np.zeros((1,2),dtype='int')[0]
    if 'cat' in name:
        y[0] = 1
    else:
        y[1] = 1
    return y

def read_images():
    for fname in random.sample(glob.glob("net_test/*.jpg"),8):
        im = Image.open(fname)
        im = (im-np.mean(im)) / np.std(im)
        label = one_hot_key(fname)
        yield im,label

def unzip(b):
    xs, ys = zip(*b)
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

model='uncompressed_Alex.pb'
test_graph = load_graph(model)
graph_input = test_graph.get_tensor_by_name('prefix/input_node:0')
graph_output = test_graph.get_tensor_by_name('prefix/output_node:0')
sess=tf.Session(graph=test_graph)
xs,ys=unzip(read_images())
predictions=np.argmax(sess.run(graph_output,feed_dict={graph_input:xs}),axis=1)
labels=np.argmax(ys,axis=1)
accuracy=100*sklearn.metrics.accuracy_score(predictions,labels)
print "For "+model+" model accuracy is: %.1f %%"%round(accuracy,2) 
tf.reset_default_graph()


# models=[]
# for file in glob.glob("models2/*.pb"):
#     models.append(file)
# TEST_KEEP_PROB=1
# for model in models:


