import cv2
import pymeanshift as pms
import os
import pickle

labels = []
for filename in os.listdir(os.getcwd()+'/images'):
	print filename
	seg_img, lbl_img, num = pms.segment(cv2.imread('images/'+filename), spatial_radius = 6, range_radius = 4.5, min_density= 50)
	with open('segs/'+filename[:-4]+'.pkl', 'w') as op:
		pickle.dump(lbl_img, op, pickle.HIGHEST_PROTOCOL)
	cv2.imwrite('labels/'+filename, seg_img)
	
for filename in os.listdir(os.getcwd()+'/tests'):
	print filename
	seg_img, lbl_img, num = pms.segment(cv2.imread('tests/'+filename), spatial_radius = 6, range_radius = 4.5, min_density= 50)
	with open('test_segs/'+filename[:-4]+'.pkl', 'w') as op:
		pickle.dump(lbl_img, op, pickle.HIGHEST_PROTOCOL)
