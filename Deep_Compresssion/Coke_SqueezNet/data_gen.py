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
	for f in glob.glob("Test1/*.jpg"):
		os.remove(f)
	for t in range(400):
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
		Image.fromarray(bg2).convert('L').point(lambda x: 0 if x<150 else 255).save('Test1/{}.jpg'.format(name))
		print t,'Done'

if __name__ == '__main__':
	Gen()
