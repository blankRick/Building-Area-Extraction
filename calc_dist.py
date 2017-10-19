import cv2
import os
import numpy as np
import operator
from collections import deque
import heapq, random
import pickle
import matplotlib.pyplot as plt

fourn = [(-1,0), (0, 1), (1,0), (0,-1)]

def dfs(visit, img, i,j,h,w ,c) :
	visit[i,j] = c;
	for x,y in fourn:
		if i+x < h and i+x > -1 and j+y < w and j+y > -1 :
			if img[i+x,j+y] != 0 and visit[i+x,j+y] == 0 :
				visit = dfs(visit, img, i+x,j+y,h,w,c)

	return visit

def dist_map(bound_seg, h, w) :
	mp = np.full((h,w), 25600, dtype=np.uint32)

	que = [(0, i, j) for i,j in bound_seg]
	for (i,j) in bound_seg :
		mp[i,j] = 0
	c=0
	while len(que) != 0:
		d,i,j = heapq.heappop(que)
		for x,y in fourn:
			if i+x < h and i+x > -1 and j+y < w and j+y > -1 :
				if mp[i+x, j+y] > d + 1 :
					mp[i+x, j+y] = d + 1
					c=c+1
					heapq.heappush(que, (mp[i+x, j+y], i+x, j+y))

	"""
	v = np.zeros((h,w), np.uint8)
	for i in range(h):
		for j in range(w) :
			v[i,j] = mp[i,j]%256
	cv2.imshow('img', v)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	"""
	print 'done %d'%c
	return mp

image_files = os.listdir(os.getcwd()+'/images')

for filename in image_files:
	img = cv2.imread('edges/'+filename, 0)
	res = cv2.imread('crf_results/'+filename, 0)

	h,w = img.shape
	visit = np.zeros((h,w), dtype = np.uint32)
	v = np.zeros((h,w), dtype = np.uint8)
	
	c=0;
	for i in range(h) :
		for j in range(w) :
			if img[i,j] != 0 and visit[i,j] == 0:
				visit = dfs(visit, img, i,j,h,w,c+1)
				c = c+1
	# for i in range(h):
	# 	for j in range(w):
	# 		v[i,j] = visit[i,j]%256
	# cv2.imshow('image', v)
	# cv2.waitKey(0)
	print c
	bound_segs = [[] for i in range(c)]
	for i in range(h) :
		for j in range(w):
			if visit[i,j] > 0:
				bound_segs[visit[i,j]-1].append((i,j))
	lens = [len(bound_segs[i]) if len(bound_segs[i])>24 else 0 for i in range(c) ]
	np.set_printoptions(threshold=np.nan)
	print lens
	#print np.bincount(np.hstack(lens))
	#break
	dist_maps = []
	bound_segs_cp = []
	for i in range(c):
		if len(bound_segs[i]) > 24 :
			dist_maps.append(dist_map(bound_segs[i],h,w))
			bound_segs_cp.append(bound_segs[i])
			# plt.imshow(dist_maps[-1])
			# plt.show()

	with open('bound_segs/'+filename[:-4]+'.pkl', 'wb') as output:
		pickle.dump(bound_segs_cp, output, pickle.HIGHEST_PROTOCOL)
	with open('distance_maps/'+filename[:-4]+'.pkl', 'wb') as output:
		pickle.dump(dist_maps, output, pickle.HIGHEST_PROTOCOL)
