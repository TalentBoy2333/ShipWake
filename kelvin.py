import numpy as np 
import cv2
import matplotlib.pyplot as plt

def create_kelvin_SAR(u=10):
	###
	# reference a Chinese paper in '/document/'
	###
	# u : velocity of ship
	G = 9.81 
	b = 4 # Half of ship width
	l = 45 # Half of ship length
	d = 7.5 # draught(depth of ship under water)
	k = G/u/u

	thetas = np.arange(-np.pi/2,np.pi/2,np.pi/180)
	x = np.arange(0,1200)
	y = np.arange(-600,600)
	x_base = np.eye(1200)
	y_base = np.eye(1200)
	for i in range(1200):
		x_base[i,:] = i
		y_base[:,i] = i-600

	temp1 = 0
	temp2 = 0
	for theta in thetas:
		print(theta)
		xy_temp1 = np.sin(k/np.cos(theta)/np.cos(theta)*((x_base+l)*np.cos(theta)+y_base*np.sin(theta)))
		xy_temp2 = np.sin(k/np.cos(theta)/np.cos(theta)*((x_base-l)*np.cos(theta)+y_base*np.sin(theta)))
		temp1 += (1-np.exp(-k*d/np.cos(theta))) * xy_temp1 * np.pi/180
		temp2 += (1-np.exp(-k*d/np.cos(theta))) * xy_temp2 * np.pi/180
		img = 4*b/np.pi/k/l*(0.6*temp1+temp2)
	# print(np.amax(img))
	# print(np.amin(img))
	cv2.imshow('img', img)
	cv2.waitKey(0)
	
	z_max = np.amax(img)-np.amin(img)
	img = (img-np.amin(img)) / z_max * 255
	img = img.astype(np.uint8)
	# print(np.amax(img))
	# print(np.amin(img))
	cv2.imwrite('kelvin.jpg', img)

def create_kelvin_simulation(u=10):
	###
	# reference 'The Speed and Beam of a Ship From Its Wake's SAR Images'
	###
	g = 9.81
	# divergent waves
	E = 1
	for i in range(10):
		n = i+1
		theta = np.arange(np.pi/180, np.arctan(np.sqrt(1/8)), np.pi/180)
		R = u**2/g * (4*np.pi*(8*n-E)*np.sin(theta)) / (3-np.sqrt(1-8*np.tan(theta)**2))**1.5 / (1+np.sqrt(1-8*np.tan(theta)**2))**0.5 / np.cos(theta)**2
		theta = theta/np.pi*180
		x,y = cv2.polarToCart(R, theta, angleInDegrees=True)
		# for j in range(len(x)):
		# 	if x[j] < 0:
		# 		x[j] = -x[j]
		# 		y[j] = -y[j]
		plt.plot(x,y,'r')
		plt.plot(x,-y,'r')
	# transverse waves
	E = 3
	for i in range(10):
		n = i+1
		theta = np.arange(np.pi/180, np.arctan(np.sqrt(1/8)), np.pi/180)
		R = u**2/g * (4*np.pi*(8*n-E)*np.sin(theta)) / (3+np.sqrt(1-8*np.tan(theta)**2))**1.5 / (1-np.sqrt(1-8*np.tan(theta)**2))**0.5 / np.cos(theta)**2
		theta = theta/np.pi*180
		x,y = cv2.polarToCart(R, theta, angleInDegrees=True)
		plt.plot(x,y,'b')
		plt.plot(x,-y,'b')
	# cusp waves
	x = np.array([0,1000])
	y = np.array([0,1000*np.sqrt(1/8)])
	plt.plot(x,y,'k')
	plt.plot(x,-y,'k')

	plt.axis([0,1000,-500,500])
	plt.show()

def cal_velocity_kelvin(Lambda):
	G = 9.81 
	v = np.sqrt(Lambda*G/(2*np.pi))
	return v

def cal_wakeLength_pixel(x1,y1,x2,y2,resolution):
	wakeLength = np.sqrt((x1-x2)**2 + (y1-y2)**2) * resolution
	return wakeLength

def cal_velocity_wakeLength(wakeLength):
	G = 9.81 
	v = np.sqrt(0.0475*wakeLength*G/(2*np.pi))
	return v

def m_s2knot(m_s):
	knot = 0.5405405405405405 * m_s * 3.6
	return knot


if __name__ == "__main__":
	create_kelvin_SAR(10)
	create_kelvin_simulation()

	# wakeLength = cal_wakeLength_pixel(832,570,676,638,1.5)
	# print('length of wave:', wakeLength)
	# v = cal_velocity_wakeLength(wakeLength)
	# print('v:', v)
	# knot = m_s2knot(v)
	# print('knot:', knot)

	# Lambda = cal_wakeLength_pixel(748,459,771,441,0.5)
	# print('Lambda:', Lambda)
	# v = cal_velocity_kelvin(Lambda)
	# print('v:', v)
	# knot = m_s2knot(v)
	# print('knot:', knot)


	# Lambda = cal_wakeLength_pixel(442,629,421,686,1)
	# print('Lambda:', Lambda)
	# v = cal_velocity_kelvin(Lambda)
	# print('v:', v)
	# knot = m_s2knot(v)
	# print('knot:', knot)