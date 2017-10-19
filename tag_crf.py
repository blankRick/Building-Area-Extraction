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


tests = os.listdir(os.getcwd()+'/tests')
test_images = [cv2.imread('tests/'+filename, 0)for filename in tests]

test_segs = [pickle.load(open('test_segs/'+filename[:-4]+'.pkl','r')) for filename in tests]

x_test = [img2features(img, seg) for img, seg in zip(test_images, test_segs)]

tagger = pcrf.Tagger()
tagger.open('building_area.crfsuite')

y_preds = [tagger.tag(test) for test in x_test]
for i in range(len(y_preds)) :
	h,w = test_images[i].shape
	res = np.zeros((h,w), dtype = np.uint8)
	for x in range(h):
		for y in range(w):
			res[x,y] = int(y_preds[i][x*w+y])
	cv2.imwrite('crf_results/'+tests[i], res)
	cv2.imshow('i', res)
	cv2.waitKey(0)
cv2.destroyAllWindows()