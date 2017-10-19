import cv2
import operator
import numpy as np
import os

def RoA(image, threshold=2.5):
	kernel_0_u1 = np.array((
		[1,1,0,0,0],
		[1,1,0,0,0],
		[1,1,0,0,0],
		[1,1,0,0,0],
		[1,1,0,0,0]),dtype="int")
	kernel_0_u2 = np.array((
		[0,0,0,1,1],
		[0,0,0,1,1],
		[0,0,0,1,1],
		[0,0,0,1,1],
		[0,0,0,1,1]),dtype="int")
	kernel_45_u1 = np.array((
		[0,0,0,0,0],
		[1,0,0,0,0],
		[1,1,0,0,0],
		[1,1,1,0,0],
		[1,1,1,1,0]),dtype="int")
	kernel_45_u2 = np.array((
		[0,1,1,1,1],
		[0,0,1,1,1],
		[0,0,0,1,1],
		[0,0,0,0,1],
		[0,0,0,0,0]),dtype="int")
	kernel_90_u1 = np.array((
		[1,1,1,1,1],
		[1,1,1,1,1],
		[0,0,0,0,0],
		[0,0,0,0,0],
		[0,0,0,0,0]),dtype="int")
	kernel_90_u2 = np.array((
		[0,0,0,0,0],
		[0,0,0,0,0],
		[0,0,0,0,0],
		[1,1,1,1,1],
		[1,1,1,1,1]),dtype="int")
	kernel_135_u1 = np.array((
		[0,0,0,0,0],
		[0,0,0,0,1],
		[0,0,0,1,1],
		[0,0,1,1,1],
		[0,1,1,1,1]),dtype="int")
	kernel_135_u2 = np.array((
		[1,1,1,1,0],
		[1,1,1,0,0],
		[1,1,0,0,0],
		[1,0,0,0,0],
		[0,0,0,0,0]),dtype="int")

	kernel_bank = (
		(kernel_0_u1, kernel_0_u2),
		(kernel_45_u1, kernel_45_u2),
		(kernel_90_u1, kernel_90_u2),
		(kernel_135_u1, kernel_135_u2))

	(ih, iw) = image.shape[:2]

	image = cv2.copyMakeBorder(image, 2,2,2,2,cv2.BORDER_REPLICATE)
	output = np.zeros((ih, iw), dtype = "float32")
	dir_out= np.zeros((ih,iw), dtype = "float32")

	for i in range(2, ih+2) :
		for j in range(2, iw+2) :
			r_val = []
			for (kernel_u1, kernel_u2) in kernel_bank :
				u1 = (image[i-2:i+2+1, j-2:j+2+1] * kernel_u1).sum()
				u2 = (image[i-2:i+2+1, j-2:j+2+1] * kernel_u2).sum()
				r_val.append(max(u1/u2, u2/u1))
			output[i-2,j-2] = max(r_val[0], r_val[1], r_val[2], r_val[3])

			if output[i-2,j-2] <= threshold :
				output[i-2,j-2] = 0
			else : 
				output[i-2, j-2] = 1.

	# for i in range(2, ih+2) :
	# 	for j in range(2, iw+2) :
	# 		r_val = []
	# 		for (kernel_u1, kernel_u2) in kernel_bank :
	# 			u1 = (image[i-2:i+2+1, j-2:j+2+1] * kernel_u1).sum()
	# 			u2 = (image[i-2:i+2+1, j-2:j+2+1] * kernel_u2).sum()
	# 			r_val.append(min(u1/u2, u2/u1))
	# 		dir_out[i-2,j-2], temp = min(enumerate(r_val), key=operator.itemgetter(1))

	return output

out = []
for filename in os.listdir(os.getcwd()+'/images'):
	image = cv2.imread('images/'+filename, 0);
	out1 = RoA(image, 2)
	out2 = cv2.GaussianBlur(image, (3,3), 0)
	lpl = cv2.Laplacian(out2,cv2.CV_64F)
	out2 = out1*lpl
	out.append(out2)
	cv2.imwrite('edges/'+filename, out2*255)
for output in out :
	cv2.imshow('image', output)
	cv2.waitKey(0)
cv2.destroyAllWindows()
