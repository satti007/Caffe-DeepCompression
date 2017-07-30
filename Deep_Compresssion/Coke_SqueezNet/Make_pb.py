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

def cod():    
    return "{}{}{}{}{}{}{} {}{}{}{}{}{}{}".format(
    random.choice(CHARS),
    random.choice(CHARS),
    random.choice(CHARS),
    random.choice(CHARS),
    random.choice(CHARS),
    random.choice(CHARS),
    random.choice(CHARS),
    random.choice(CHARS),
    random.choice(CHARS),
    random.choice(CHARS),
    random.choice(CHARS),
    random.choice(CHARS),
    random.choice(CHARS),
    random.choice(CHARS))


def Gen():
	for f in glob.glob("Im/*.jpg"):
		os.remove(f)
	for t in range(32):
		bg = Image.new('RGBA', (400,200), (255,255,255,255))
		im = bg.copy();para = textwrap.wrap(cod(), width=7)
		d = ImageDraw.Draw(im)
		font = random.choice(['c','d','e','f', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
		if font == 'a':
			fa = ImageFont.truetype('fonts/a.ttf', randint(59,61)); font = fa
			a = (randint(20,60),randint(20,40)); b = (randint(20,60),randint(100,140))
		elif font == 'c':
			fc = ImageFont.truetype('fonts/c.ttf', randint(66,68)); font = fc
			a = (randint(20,60),randint(-20,0)); b = (randint(20,60),randint(60,100))
		elif font == 'd':
			fd = ImageFont.truetype('fonts/d.ttf', randint(66,68)); font = fd
			a = (randint(20,60),randint(20,40)); b = (randint(20,60),randint(100,140))
		elif font == 'e':
			fe = ImageFont.truetype('fonts/e.ttf', randint(59,61)); font = fe
			a = (randint(20,60),randint(20,40)); b = (randint(20,60),randint(100,140))
		elif font == 'f':
			ff = ImageFont.truetype('fonts/f.ttf', randint(119,121)); font = ff
			a = (randint(20,60),randint(0,20)); b = (randint(20,60),randint(80,120))
		elif int(font) <= 6:
			fn1 = ImageFont.truetype('fonts/{}.ttf'.format(font), randint(67,69));font=fn1;
			a = (randint(20,60),randint(20,40)); b = (randint(20,60),randint(100,140))
		elif int(font) > 6:
			fn2 = ImageFont.truetype('fonts/{}.ttf'.format(font), randint(97,99));font=fn2;
			a = (randint(20,60),randint(20,40)); b = (randint(20,60),randint(100,140))
		para = textwrap.wrap(cod(),7); name=para[0]+para[1]
		d.text(a, para[0], font=font, fill='black')
		d.text(b, para[1], font=font, fill='black')
		a1 = randint(150,230); b1 = int(a1/2)+randint(12,20);
		x1 = randint(int(a1/5)-10,int(a1/5)+10); y1 = randint(45,100)
		im = im.rotate(angle = randint(-12,12), resample=Image.BICUBIC, expand=True).resize((a1,b1), Image.ANTIALIAS).convert('RGBA')
		bg1= Image.new('RGBA', (256,256), (255,255,255,255));
		bg1.paste(im, (x1,y1), im); 
		p = np.ones((2,2));
		bg2 = cv2.erode(np.array(bg1), p, iterations=0);
		Image.fromarray(bg2).convert('L').point(lambda x: 0 if x<150 else 255).save('Im/{}.jpg'.format(name))




## ****** Train ****** ##

def code_to_vec(code):
    def char_to_vec(c):
        y = np.zeros((len(CHARS),))
        y[CHARS.index(c)] = 1.0
        return y
    c = np.vstack([char_to_vec(c) for c in code])
    return c.flatten()


def unzip(b):
    xs, ys = zip(*b)
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys


def read_data_train():
    Gen()
    for fname in (glob.glob("Im/*.jpg")):
	im = Image.open(fname).convert('L').point(lambda x: 0 if x<150 else 255);
	im1 = np.array(im)/ 255.; code = fname.split("/")[-1][:-4]
        yield im1, code_to_vec(code)


def read_data_test():
    for fname in (glob.glob("Test/*.jpg")[250:400]):
	im = Image.open(fname).convert('L').point(lambda x: 0 if x<150 else 255);
	im1 = np.array(im)/ 255.; code = fname.split("/")[-1][:-4]
        yield im1, code_to_vec(code)


len_test = len(unzip(read_data_test())[0])


def get_loss(y, y_):
    digits_loss = tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(y[:, :],[-1, len(CHARS)]),labels=tf.reshape(y_[:, :],[-1, len(CHARS)]))
    digits_loss = tf.reshape(digits_loss, [-1, 14])
    digits_loss = tf.reduce_sum(digits_loss, 1)
    digits_loss = tf.reduce_sum(digits_loss)
    return digits_loss



def train(learn_rate, initial_weights=None):
    x, y, params = New_Model_Orig.build_model_f()
    y_ = tf.placeholder(tf.float32, [None, 14 * len(CHARS)])
    digits_loss = get_loss(y, y_)
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(digits_loss)
    best = tf.argmax(tf.reshape(y, [-1, 14, len(CHARS)]), 2)
    correct = tf.argmax(tf.reshape(y_, [-1, 14, len(CHARS)]), 2)
    if initial_weights is not None:
        assert len(params) == len(initial_weights)
        assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)]
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=2)
    tf.add_to_collection('input_node', x)
    tf.add_to_collection('output_node', y)
    def vec_to_plate(v):
        return "".join(CHARS[i] for i in v)
    def do_report(batch_idx):
        r = sess.run([best,
                      correct,
                      tf.greater(y[:, 0], 0),
                      y_[:, 0],
                      digits_loss],
                     feed_dict={x: test_xs, y_: test_ys})
        r_short = (r[0][:35], r[1][:35], r[2][:35], r[3][:35])
        for b, c, pb, pc in zip(*r_short):
	    vc = vec_to_plate(c);vb = vec_to_plate(b)
            print "{} <->  {} <-> {}".format(vc,vb,np.sum(a==b for a, b in zip(vc, vb)))
        #num_p_correct = np.sum(r[2] == r[3])
	Matches = np.sum(np.all(r[0] == r[1], axis=1))
	comp = (r[0] == r[1]);sum = 0;index = []
	for i in range(len(r[1][:,0])):
		if r[1][:,0][i] > 0:
			sum = sum + np.sum(comp[i])
			index.append(i)
	Char_Acc = (sum * 100.)/(14 * len(index))
        print ("\n *****Squeeze_with_BN***** \nBatch-{:3d}; \nCharacter_Accuracy: {:2.02f}%; \nMatching_Accuracy: {:02.02f}% \nCharacter_loss: {}; \nTotal time: {} hrs").format(
            batch_idx,
            Char_Acc,
	    100. * Matches / len(r[2]),
            r[4],(time.time() - t0)/3600.); f.flush(); os.fsync(f.fileno()); writer1.writerow([batch_idx, Char_Acc, 100. * Matches / len(r[2]), r[4],round((time.time() - t0)/3600.,3)])
    def do_batch():
        sess.run(train_step,feed_dict={x: train_xs, y_: train_ys})    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    with tf.Session(config=config) as sess:
        sess.run(init)
        if initial_weights is not None:
            sess.run(assign_ops)          
        test_xs, test_ys = unzip(list(read_data_test())[:200])
        tf.train.write_graph(sess.graph_def, 'Model/', 'graph.pb', False)
        writer = tf.summary.FileWriter('Model', sess.graph)
        from tensorflow.python.framework.graph_util import convert_variables_to_constants
        minimal_graph = convert_variables_to_constants(sess, sess.graph_def, ["output_node"])
        tf.train.write_graph(minimal_graph, '.', 'Squeeze_frozen_graph.pb', as_text=False)
        
        with open("Squeeze_Results.csv",'wb') as f:
          writer1 = csv.writer(f, dialect='excel')
          batch_idx = 0
          while True:
            batch_idx = batch_idx + 1; t1 = time.time()
            train_xs, train_ys = unzip(list(read_data_train()))
            do_batch();
            if batch_idx % 10 == 0:
               do_report(batch_idx) 
            if batch_idx % 200 == 0:
               last_weights = [p.eval() for p in params]
               np.savez("Model/Squeeze_Weights.npz", *last_weights)
               minimal_graph_2 = convert_variables_to_constants(sess, sess.graph_def, ["output_node"])
               tf.train.write_graph(minimal_graph_2, '.', 'Squeeze_Test.pb', as_text=False)

f = np.load('Latest_Sq_Wts_100_100_18_Apr.npz')
initial_weights = [f[p] for p in sorted(f.files,key=lambda s: int(s[4:]))]
train(1e-3,initial_weights=initial_weights)
