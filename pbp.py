import cv2
import os
import numpy as np
import operator
from collections import deque
import heapq, random
import pickle
import matplotlib.pyplot as plt

fourn = [(-1,0), (0, 1), (1,0), (0,-1)]
RN = 15

def find_s(crf_res, dist_map, bound_seg, dist, h , w) :
	S0 = 0
	S1 = 0
	visit = np.zeros(dist_map.shape, dtype = np.uint8)
	S_vals = [[0,0] for i in range(dist+1)]
	que = [(0, i, j) for i,j in bound_seg]
	for i,j in bound_seg :
		visit[i,j] = 1
	#v = np.zeros((h,w), np.uint8)
	while len(que) != 0:
		# print len(que)
		d,i,j = que.pop()
		for x,y in fourn:
			if i+x < h and i+x > -1 and j+y < w and j+y > -1 :
				if dist_map[i+x, j+y] <= dist and visit[i+x,j+y] == 0:
					if(crf_res[i+x,j+y]==0) :
						S_vals[dist_map[i+x, j+y]][0] = S_vals[dist_map[i+x, j+y]][0]+1
					else :
						S_vals[dist_map[i+x, j+y]][1] = S_vals[dist_map[i+x, j+y]][1]+1
					visit[i+x, j+y] = 1
					# print 'jere'
					que.append((dist_map[i+x,j+y], i+x, j+y))

	# cv2.imshow('img', v)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# plt.imshow(dist_map)
	# plt.show()
	print S_vals
	return S_vals

def find_t(S0, S1, c, dist) :
	if dist <= RN :
		if c==0 :
			if S0 > S1:
				return 1
			else:
				return -1
		else :
			if S1 >= S0:
				return 1
			else:
				return -1
	else :
		return -1

def find_st(crf_res, dist_map, bound_seg, dist, h , w) :
	S0 = 0
	S1 = 0
	visit = np.zeros(dist_map.shape, dtype = np.uint8)
	que = [(0, i, j) for i,j in bound_seg]
	for i,j in bound_seg :
		visit[i,j] = 1
	#v = np.zeros((h,w), np.uint8)
	while len(que) != 0:
		d,i,j = que.pop()
		for x,y in fourn:
			if i+x < h and i+x > -1 and j+y < w and j+y > -1 :
				if dist_map[i+x, j+y] <= dist and visit[i+x,j+y] == 0:
					if(crf_res[i+x,j+y]==0) :
						S0 = S0+1
					else :
						S1 = S1+1
					visit[i+x, j+y] = 1
					que.append((dist_map[i+x,j+y], i+x, j+y))

	if S0 >= S1:
		return 1, -1
	else:
		return -1, 1

image_files = os.listdir(os.getcwd()+'/crf_results')
first = True
for filename in image_files:

	res = cv2.imread('crf_results/'+filename, 0)
	# print np.bincount(np.hstack(res))
	bound_segs = pickle.load(open('bound_segs/'+filename[:-4]+'.pkl', 'rb'))
	dist_maps = pickle.load(open('distance_maps/'+filename[:-4]+'.pkl', 'rb'))
	N = 3
	h,w = res.shape
	fin_res = np.zeros((h,w), np.uint8)
	max_val = max(h,w)
	S_vals = [find_s(res, dist_map, bound_segs, RN, h, w) for dist_map, bound_segs in zip(dist_maps, bound_segs)]
	print 'found Svals'
	for x in range(h) :
		for y in range(w) :
			all_omega = [dist_maps[j][x,y] for j in range(len(dist_maps))]
			
			omega = heapq.nsmallest(N, enumerate(all_omega), key=lambda s:s[1])
			# if (res[x,y] != 0) and not first :
			# 	print x,y, res[x,y]
			# 	print [S_vals[j] for j, dist in omega]
			# 	for j, dist in omega :
			# 		plt.imshow(dist_maps[j])
			# 		plt.show()
			
			# t_pi_R = [find_st(res, dist_maps[j], bound_segs[j], min(dist_maps[j][x,y], 15), h, w) for j,dist in omega]
			S=[]
			for j, dist in omega:
				print [S_val[0] for S_val in S_vals[j]]
				if res[x,y] != 0:
					S0 = sum([S_val[0] for S_val in S_vals[j]][:min(dist+1, RN+1)])
					S1 = sum([S_val[1] for S_val in S_vals[j]][:min(dist+1, RN+1)])
				else:				
					S0 = sum([S_val[0] for S_val in S_vals[j]][:RN+1])
					S1 = sum([S_val[1] for S_val in S_vals[j]][:RN+1])
				S.append((S0, S1, dist))
			print S
			t_pi_R = [(find_t(S0, S1, 0, dist), find_t(S0, S1, 1, dist)) for S0, S1, dist in S]
			# if(res[x,y] != 0) and not first:
			# 	print t_0_pi_R
			# 	print t_1_pi_R
			F_dist_pi_oj = [1./(1.+2.71828**(-(dist/max_val))) for j, dist in omega]

			P_0 = sum([0.9*t0*f for (t0, t1),f in zip(t_pi_R, F_dist_pi_oj)])
			P_1 = sum([t1*f for (t0, t1),f in zip(t_pi_R, F_dist_pi_oj)])

			if P_0 >P_1 :
				fin_res[x,y] = 0
			else :
				#print 'yes'
				fin_res[x,y] = 255
			
		print x

	cv2.imwrite('fin_results/'+filename, fin_res)
	cv2.imshow('img', fin_res)
	cv2.waitKey(0)
	first = False
	print 'done'

cv2.destroyAllWindows()
