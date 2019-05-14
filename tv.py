import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import math

def total_variation(imageName, Lambda=0.2, a=0.5, iterNum=300, show=False):
	img = cv2.imread(imageName)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(img, (3,3), 0)
	img = img.astype(np.float)
	m,n = img.shape
	# print(m,n)
	if show:
		plt.subplot(121), plt.imshow(img, 'gray')
	imgNew = img
	subplotIndex = 331
	for l in range(iterNum):
		if (l+1) % 10 == 0 and l != 0:
			print('total variation iter', l+1)
		# if l%50 == 0:
		# 	plt.subplot(subplotIndex), plt.imshow(imgNew, 'gray')
		# 	subplotIndex += 1

		Un_img1, Un_img2, Un_img3, Un_img4 = img[1:m-1,1:n-1], img[0:m-2,1:n-1], img[0:m-2,0:n-2], img[0:m-2,2:n]
		Ue_img1, Ue_img2, Ue_img3, Ue_img4 = img[1:m-1,1:n-1], img[1:m-1,2:n], img[0:m-2,2:n], img[2:m,2:n]
		Uw_img1, Uw_img2, Uw_img3, Uw_img4 = img[1:m-1,1:n-1], img[1:m-1,0:n-2], img[0:m-2,0:n-2], img[2:m,0:n-2]
		Us_img1, Us_img2, Us_img3, Us_img4 = img[1:m-1,1:n-1], img[2:m,1:n-1], img[2:m,0:n-2], img[2:m,2:n]

		Un = np.sqrt((Un_img1-Un_img2)**2 + ((Un_img3-Un_img4)/2)**2)
		Ue = np.sqrt((Ue_img1-Ue_img2)**2 + ((Ue_img3-Ue_img4)/2)**2)
		Uw = np.sqrt((Uw_img1-Uw_img2)**2 + ((Uw_img3-Uw_img4)/2)**2)
		Us = np.sqrt((Us_img1-Us_img2)**2 + ((Us_img3-Us_img4)/2)**2)

		Wn = 1 / np.sqrt(Un**2+a**2)
		We = 1 / np.sqrt(Ue**2+a**2)
		Ww = 1 / np.sqrt(Uw**2+a**2)
		Ws = 1 / np.sqrt(Us**2+a**2)

		Hon = Wn / (Wn + We + Ww + Ws + Lambda)
		Hoe = We / (Wn + We + Ww + Ws + Lambda)
		How = Ww / (Wn + We + Ww + Ws + Lambda)
		Hos = Ws / (Wn + We + Ww + Ws + Lambda)

		Hoo = Lambda / (Wn + We + Ww + Ws + Lambda)
		imgNew[1:m-1,1:n-1] = Hon*img[0:m-2,1:n-1] + Hoe*img[1:m-1,2:n] + How*img[1:m-1,0:n-2] + Hos*img[2:m,1:n-1] + Hoo*img[1:m-1,1:n-1]

		# for i in np.arange(1,m-2):
		# 	for j in np.arange(1,n-2):
		# 		Un = np.sqrt((img[i,j]-img[i-1,j])**2 + ((img[i-1,j-1]-img[i-1,j+1])/2)**2)
		# 		Ue = np.sqrt((img[i,j]-img[i,j+1])**2 + ((img[i-1,j+1]-img[i+1,j+1])/2)**2)
		# 		Uw = np.sqrt((img[i,j]-img[i,j-1])**2 + ((img[i-1,j-1]-img[i+1,j-1])/2)**2)
		# 		Us = np.sqrt((img[i,j]-img[i+1,j])**2 + ((img[i+1,j-1]-img[i+1,j+1])/2)**2)

		# 		Wn = 1 / np.sqrt(Un**2+a**2)
		# 		We = 1 / np.sqrt(Ue**2+a**2)
		# 		Ww = 1 / np.sqrt(Uw**2+a**2)
		# 		Ws = 1 / np.sqrt(Us**2+a**2)

		# 		Hon = Wn / (Wn + We + Ww + Ws + Lambda)
		# 		Hoe = We / (Wn + We + Ww + Ws + Lambda)
		# 		How = Ww / (Wn + We + Ww + Ws + Lambda)
		# 		Hos = Ws / (Wn + We + Ww + Ws + Lambda)

		# 		Hoo = Lambda / (Wn + We + Ww + Ws + Lambda)

		# 		imgNew[i,j] = Hon*img[i-1,j] + Hoe*img[i,j+1] + How*img[i,j-1] + Hos*img[i+1,j] + Hoo*img[i,j]
		img = imgNew

	if show:
		plt.subplot(122), plt.imshow(imgNew, 'gray')
		plt.show()

	return imgNew



if __name__ == "__main__":
	imgTV = total_variation('image/test.png', 0.2, 0.05, 100)
	cv2.imwrite('result/tv_result.png', imgTV)
