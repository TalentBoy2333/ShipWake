import sys
from tv import *
from wake_detection import *
from kelvin import *

def main(imageName):
	# total variation
	imgTV = total_variation(imageName, 0.2, 0.05, 100)
	tvName = 'result/tv_result.png'
	cv2.imwrite(tvName, imgTV)
	# hough transform
	lines = get_wake_lines(tvName, show=False)
	maxLine = max_lines(lines)
	draw_line(imageName, maxLine)
	# slopeBox = box_slope(maxLine, 30)
	velocity = cal_velocity(maxLine, 1.5)
	# draw_bboxs(imageName, slopeBox)
	knot = m_s2knot(velocity)
	velocity = np.array('%.3f'%velocity + 'm/s  ' + str(round(knot)) + 'knot')
	print(velocity)
	np.save('result/shipVelocity.npy', velocity)



if __name__ == "__main__":
	imageName = sys.argv[1]
	print(imageName)
	main(imageName)

