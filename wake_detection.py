import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_wake_lines(imageName, show=False):
	img = cv2.imread(imageName) 
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# gray = cv2.GaussianBlur(gray, (3,3), 0)
	if show:
		plt.subplot(131),plt.imshow(gray,'gray')
		plt.xticks([]),plt.yticks([])
	# canny
	edges = cv2.Canny(gray, 5, 150, apertureSize = 3)
	kernel = np.ones((5,5), np.uint8) 
	edges = cv2.dilate(edges, kernel, iterations = 1)
	# edges = np.zeros((gray.shape[0],gray.shape[1]))
	# edges[gray>=50] = 255
	# edges[gray<50] = 0
	# edges = edges.astype(np.uint8)
	if show:
		plt.subplot(132),plt.imshow(edges,'gray')
		plt.xticks([]),plt.yticks([])
	#hough transform
	lines = cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength=20,maxLineGap=30)
	# lines = cv2.HoughLines(edges,1,np.pi/180,10)
	# print(lines)
	if show:
		lines1 = lines[:,0,:]
		for x1,y1,x2,y2 in lines1[:]:
			# print(x1,y1,x2,y2) 
			cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)
		plt.subplot(133),plt.imshow(img,)
		plt.xticks([]),plt.yticks([])
		plt.show()
		cv2.imwrite('result/hough_result.png', img)

	return lines[:,0,:]

def nms_lines(lines):
	new_lines = list()
	for x1,y1,x2,y2 in lines:
		new_x1 = min(x1,x2)
		new_x2 = max(x1,x2)
		new_y1 = min(y1,y2)
		new_y2 = max(y1,y2)
		new_lines.append([new_x1,new_y1,new_x2,new_y2])
	boxs = []
	for x1,y1,x2,y2 in new_lines:
		s_line = (x2-x1)*(y2-y1)
		flag = True
		for i in range(len(boxs)):
			x3,y3,x4,y4 = boxs[i]
			s_box = (x4-x3)*(y4-y3)
			s_overlap = (min(x2,x4)-max(x1,x3))*(min(y2,y4)-max(y1,y3))
			s_all = s_line + s_box - s_overlap
			if s_overlap/s_all >= 0 and min(x2,x4) >= max(x1,x3) and min(y2,y4) >= max(y1,y3):
				flag = False
				if s_line > s_box:
					boxs[i] = [x1,y1,x2,y2]
				break
		if flag:
			boxs.append([x1,y1,x2,y2])
	return boxs

def max_lines(lines):
	s_maxLine = 0
	for x1,y1,x2,y2 in lines:
		s_line = abs(x2-x1) * abs(y2-y1)
		if s_line > s_maxLine:
			maxLine = [x1,y1,x2,y2]
			s_maxLine = s_line
	return maxLine

def draw_line(imageName, maxLine):
	img = cv2.imread(imageName)
	x1,y1,x2,y2 = maxLine
	cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,0), 2) 
	cv2.imwrite('result/result.png', img)
	return img

def box_slope(maxLine, width):
	x1,y1,x2,y2 = maxLine
	# print('x1,y1,x2,y2:', x1,y1,x2,y2)
	k = (y2-y1) / (x2-x1)
	# print('k:',k)
	theta = np.arctan(k)
	delta_x = abs(int(width*np.sin(theta)))
	delta_y = abs(int(width*np.cos(theta)))
	if k >= 0:
		slope1_x = x1 - delta_x
		slope1_y = y1 + delta_y
		slope2_x = x1 + delta_x
		slope2_y = y1 - delta_y
		slope3_x = x2 + delta_x
		slope3_y = y2 - delta_y
		slope4_x = x2 - delta_x
		slope4_y = y2 + delta_y
	elif k < 0:
		slope1_x = x1 - delta_x
		slope1_y = y1 - delta_y
		slope2_x = x1 + delta_x
		slope2_y = y1 + delta_y
		slope3_x = x2 + delta_x
		slope3_y = y2 + delta_y
		slope4_x = x2 - delta_x
		slope4_y = y2 - delta_y
	slopeBox = [slope1_x, slope1_y, slope2_x, slope2_y, slope3_x, slope3_y, slope4_x, slope4_y]
	return slopeBox

def cal_velocity(maxLine, resolutioon):
	x1,y1,x2,y2 = maxLine
	pixelNum = np.sqrt((x1-x2)**2+(y1-y2)**2)
	wakeLength = pixelNum * resolutioon
	velocity = np.sqrt((0.06*wakeLength*9.81) / (2*np.pi))
	return velocity

def draw_bboxs(imageName, slopeBox):
	img = cv2.imread(imageName)
	cv2.line(img,(slopeBox[0],slopeBox[1]),(slopeBox[2],slopeBox[3]),(255,255,0),2)
	cv2.line(img,(slopeBox[2],slopeBox[3]),(slopeBox[4],slopeBox[5]),(255,255,0),2)
	cv2.line(img,(slopeBox[4],slopeBox[5]),(slopeBox[6],slopeBox[7]),(255,255,0),2)
	cv2.line(img,(slopeBox[6],slopeBox[7]),(slopeBox[0],slopeBox[1]),(255,255,0),2)
	
	cv2.imwrite('result/result.png', img)
	return img

# def cal_velocity(boxs, resolutioon):
# 	weaks = list()
# 	for x1,y1,x2,y2 in boxs:
# 		pixelNum = np.sqrt((x1-x2)**2+(y1-y2)**2)
# 		wakeLength = pixelNum * resolutioon
# 		velocity = np.sqrt((0.0475*wakeLength*9.81) / (2*np.pi))
# 		weaks.append([x1,y1,x2,y2,wakeLength,velocity])
# 	return weaks

# def draw_bboxs(imageName, weaks):
# 	img = cv2.imread(imageName)
# 	for x1,y1,x2,y2,wakeLength,velocity in weaks:
# 		cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,0), 2) 
# 		# font = cv2.FONT_HERSHEY_SIMPLEX
# 		# text = 'l:'+str('%.2f'%wakeLength)+'v:'+str('%.3f'%velocity)
# 		# cv2.putText(img, text, (x1,y1-5), font, 1, (255,255,0), 1)
# 	# cv2.imshow('img', img)
# 	# cv2.waitKey(0)
# 	cv2.imwrite('result/result.png', img)
# 	return velocity

			





if __name__ == "__main__":
	# imageName = 'image/test.png'
	imageName = 'result/tv_result.png'
	lines = get_wake_lines(imageName, show=True)
	maxLine = max_lines(lines)
	slopeBox = box_slope(maxLine, 30)
	velocity = cal_velocity(maxLine, 1.5)
	print(velocity)
	draw_bboxs(imageName, slopeBox)


	
