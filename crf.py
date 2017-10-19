import pycrfsuite as pcrf
import os
import cv2
import pickle
import numpy as np

def pixel2features(img, seg, i, j, h, w) :
	features = [
		'bias',
		'row=' + str(float(i)/float(h)),
		'col=' + str(float(j)/float(w)),
		'color=' + str(img[i,j]),
		'seg_num=' + str(seg[i,j]),
	]
	if i>0:
		features.extend([
			'-10:row=' + str(float(i-1)/float(h)),
			'-10:col=' + str(float(j)/float(w)),
			'-10:color=' + str(img[i-1,j]),
			'-10:seg_num=' + str(seg[i-1,j]),
		])
	else :
		features.append('L')
	if i<h-1:
		features.extend([
			'+10:row=' + str(float(i+1)/float(h)),
			'+10:col=' + str(float(j)/float(w)),
			'+10:color=' + str(img[i+1,j]),
			'+10:seg_num=' + str(seg[i+1,j]),
		])
	else :
		features.append('R')
	if j>0:
		features.extend([
			'0-1:row=' + str(float(i)/float(h)),
			'0-1:col=' + str(float(j-1)/float(w)),
			'0-1:color=' + str(img[i,j-1]),
			'0-1:seg_num=' + str(seg[i,j-1]),
		])
	else :
		features.append('T')
	if j<w-1:
		features.extend([
			'0+1:row=' + str(float(i)/float(h)),
			'0+1:col=' + str(float(j+1)/float(w)),
			'0+1:color=' + str(img[i,j+1]),
			'0+1:seg_num=' + str(seg[i,j+1]),
		])
	else :
		features.append('B')
	return features


def img2features(img, seg) :
	h,w = img.shape[:2]
	return [pixel2features(img, seg, i, j, h, w) for i in range(h) for j in range(w)]

def img2labels(img) :
	h,w = img.shape[:2]
	return [str(img[i,j]) for i in range(h) for j in range(w)]

image_files = os.listdir(os.getcwd()+'/images')
train_images = [cv2.imread('images/'+filename, 0)for filename in image_files]
label_images = [cv2.imread('labels/'+filename, 0) for filename in image_files]
segmt_images = [pickle.load(open('segs/'+filename[:-4]+'.pkl','r')) for filename in image_files]

x_train = [img2features(img, seg) for img, seg in zip(train_images, segmt_images)]
y_train = [img2labels(img) for img in label_images]

trainer = pcrf.Trainer(verbose=False)

for xseq, yseq in zip(x_train, y_train):
	trainer.append(xseq, yseq)

trainer.set_params({
	'c1' : 1.0,
	'c2' : 1e-3,
	'max_iterations' : 10,
	'feature.possible_transitions':True
	})

trainer.train('building_area.crfsuite')
