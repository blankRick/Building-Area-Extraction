import pycrfsuite as pcrf
import os
import cv2
import pickle
import numpy as np
from pystruct import learners
import pystruct.models as crfs
from pystruct.utils import SaveLogger

def img2edges(seg) :
	h,w = seg.shape[:2]
	edges=[]
	edge_set = set()
	for i in range(h-1) : 
		for j in range(w-1) : 
			if seg[i,j]!=seg[i,j+1] :
				if (seg[i,j], seg[i,j+1]) not in edge_set or (seg[i,j+1], seg[i,j]) not in edge_set:
					edge_set.add((seg[i,j], seg[i,j+1]))
					edge_set.add((seg[i,j+1], seg[i,j]))
					edges.append([seg[i,j], seg[i,j+1]])
			if seg[i+1,j]!=seg[i,j] :
				if (seg[i,j], seg[i+1,j]) not in edge_set or (seg[i+1,j], seg[i,j]) not in edge_set:
					edge_set.add((seg[i,j], seg[i+1,j]))
					edge_set.add((seg[i+1,j], seg[i,j]))
					edges.append([seg[i,j], seg[i+1,j]])
	return np.array(edges, dtype=int)

def img2labels(lbl, seg) :
	h,w = lbl.shape[:2]
	countSegs = np.bincount(np.hstack(seg)).shape[0]
	print countSegs
	yt = [0 for i in range(countSegs)]
	for i in range(h):
		for j in range(w) :
			if lbl[i,j] == 255 : 
				yt[seg[i,j]] = 1
			else :
				yt[seg[i,j]] = 0

	return np.array(yt)

def img2attrbs(lbl, seg) :
	h,w = lbl.shape[:2]
	countSegs = np.bincount(np.hstack(seg)).shape[0]
	print countSegs
	xt = [[0 for i in range(2)] for j in range(countSegs)]
	for i in range(h):
		for j in range(w) :
			if lbl[i,j] == 255 : 
				xt[seg[i,j]][0] = 1
			else :
				xt[seg[i,j]][1] = 1

	return np.array(xt, dtype=np.float32)

tests = os.listdir(os.getcwd()+'/tests')
test_images = [cv2.imread('tests/'+filename, 0)for filename in tests]

test_segs = [pickle.load(open('test_segs/'+filename[:-4]+'.pkl','r')) for filename in tests]

edges_train = [img2edges(seg) for seg in test_segs]
attrbs_train = [img2attrbs(lbl,seg) for lbl, seg in zip(label_images, test_segs)]

image_files = os.listdir(os.getcwd()+'/images')
train_images = [cv2.imread('images/'+filename, 0)for filename in image_files]
label_images = [cv2.imread('labels/'+filename, 0) for filename in image_files]

segmt_images = [pickle.load(open('segs/'+filename[:-4]+'.pkl','r')) for filename in image_files]

edges_train = [img2edges(seg) for seg in segmt_images]
attrbs_train = [img2attrbs(lbl,seg) for lbl, seg in zip(label_images, segmt_images)]

x_train = [(xt, edges) for xt, edges in zip(attrbs_train, edges_train)]
y_train = [img2labels(lbl, seg) for lbl, seg in zip(label_images, segmt_images)]

print 'prepared'
C = 0.01

n_states = 2
print("number of samples: %s" % len(x_train))
class_weights = 1. / np.bincount(np.hstack(y_train))
class_weights *= 2. / np.sum(class_weights)
print(class_weights)

print>>open('out.txt', 'w'), x_train

model = crfs.GraphCRF(inference_method='ad3', class_weight=class_weights)

experiment_name = "edge_features_one_slack_trainval_%f" % C

ssvm = learners.NSlackSSVM(
    model, verbose=2, C=C, max_iter=100000, n_jobs=-1,
    tol=0.0001, show_loss_every=5,
    logger=SaveLogger(experiment_name + ".pickle", save_every=100),
    inactive_threshold=1e-3, inactive_window=10, batch_size=100)
ssvm.fit(x_train, y_train)

x_test = []
y_pred = ssvm.predict(x_train)

# we throw away void superpixels and flatten everything
y_pred, y_true = np.hstack(y_pred), np.hstack(y_train)
y_pred = y_pred[y_true != 255]
y_true = y_true[y_true != 255]

print("Score on validation set: %f" % np.mean(y_true == y_pred))