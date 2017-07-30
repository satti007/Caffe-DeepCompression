import time
t0 = time.time()
import math
import sys
import random
import os
import cv2,csv
import glob
import numpy as np
import textwrap
import tensorflow as tf
import New_Model_Orig

from random import randint
from PIL import Image, ImageDraw, ImageFont, ImageFilter
DIGITS = "456790"
LETTERS = "BFHJKLMNPRTVWX"
CHARS = LETTERS + DIGITS

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def unzip(b):
    xs, ys = zip(*b)
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys



def code_to_vec(code):
    def char_to_vec(c):
        y = np.zeros((len(CHARS),))
        y[CHARS.index(c)] = 1.0
        return y
    c = np.vstack([char_to_vec(c) for c in code])
    return c.flatten()


def vec_to_plate(v):
  return "".join(CHARS[i] for i in v)


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

Sq_graph = load_graph("Squeeze_Test.pb")
Sq_input = Sq_graph.get_tensor_by_name('prefix/input_node:0')
Sq_output = Sq_graph.get_tensor_by_name('prefix/output_node:0')
s = tf.Session(graph=Sq_graph); 

Matches=[]
Orig = []
Pred = []
for f in glob.glob("Test/*"):
  try: 
    im = Image.open(f).convert('L').point(lambda x: 0 if x<150 else 255); im1 = np.array(im)/ 255.;
    ctv = code_to_vec(f.split('/')[-1][:14])
    p = s.run(Sq_output,feed_dict={Sq_input: im1.reshape([1,256,256])})
    pp = np.argmax(p.reshape([14,20]), 1); qq = np.argmax(ctv.reshape([14,20]), 1)
    Matches.append(np.sum(pp == qq)); Orig.append(vec_to_plate(qq)); Pred.append(vec_to_plate(pp))
  except:
    print f.split('/')[-1][:-4]


for i in range(len(Pred)):
  print "{} <------> {} <------> {}".format(Orig[i],Pred[i],Matches[i])


print 'Final_Char_Acc: {}'.format(np.sum(Matches)/(len(Matches)*14.))
print 'Final_Bingo_Acc: {}'.format(sum([1 for i in Matches if i==14])/(len(Matches)*1.))
print 'One-Away Acc: {}'.format(sum([1 for i in Matches if i>=13])/(len(Matches)*1.))



























