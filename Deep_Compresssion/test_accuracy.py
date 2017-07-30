import os
import glob
import numpy as np
import sklearn.metrics
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

mnist = input_data.read_data_sets('MNIST_data', one_hot=True,reshape=False)
index=[]
file=open('index.txt','r')
for line in file:
    index.append(int(line.strip('\n')))

file.close()
test_images=np.zeros([1000,28,28,1])
test_labels=np.zeros([1000,10])
count=0
for i in index:
	test_images[count]=mnist.test.images[i][:][:][:]
	test_labels[count]=mnist.test.labels[i][:]
	count=count+1

# models=[]
# for file in glob.glob("models2/*.pb"):
#     models.append(file)

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

TEST_KEEP_PROB=1

# print ''
# for model in models:
	# A=os.system('gunzip '+model+'.gz')
model='uncompressed_MNIST.pb'
MNIST_graph = load_graph(model)
graph_input = MNIST_graph.get_tensor_by_name('prefix/input_node:0')
graph_output = MNIST_graph.get_tensor_by_name('prefix/output_node:0')
graph_dropout1 = MNIST_graph.get_tensor_by_name('prefix/keep_prob1:0')
graph_dropout2 = MNIST_graph.get_tensor_by_name('prefix/keep_prob2:0')
sess=tf.Session(graph=MNIST_graph)
feed_dict = {graph_input:test_images,graph_dropout1: TEST_KEEP_PROB, graph_dropout2: TEST_KEEP_PROB}
predictions=np.argmax(sess.run(graph_output,feed_dict=feed_dict),axis=1)
labels=np.argmax(test_labels,axis=1)
acc=100*sklearn.metrics.accuracy_score(predictions,labels)
print "For "+model+" model accuracy is: %.1f %%"%round(acc,2) 
tf.reset_default_graph()


# print ''
# for model in models:
# 	# A=os.system('gzip '+model)
# 	# A=os.system('ls '+model+'.gz'+' -lh')
# 	A=os.system('ls '+model+' -lh')

