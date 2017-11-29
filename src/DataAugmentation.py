import cv2
import numpy as np
import settings.DataAugmentSettings as dataAugmentSettings

class DataAugmentation:
	def __init__(self):
		pass

	def _HorizontalFlip(self, image, u=0.5, v=1.0):   
		if v < u:
			image = cv2.flip(image, 1)
		return image

	def _VerticalFlip(self, image, u=0.5, v=1.0):
		if v < u:
			image = cv2.flip(image, 0)
		return image

	def Flip(self, image):
		image = self._HorizontalFlip(image, u=0.5, v=np.random.random())
		image = self._VerticalFlip(image, u=0.5, v=np.random.random())
		return image

	def _HorizontalShift(self, image, u=0.33, v=1.0):
		rightShift = np.float32([[1,0,15],[0,1,0]])
		leftShift = np.float32([[1,0,-15],[0,1,0]])
		rows,cols = image.shape[:2]
		if v < u:
			image = cv2.warpAffine(image, rightShift, (rows, cols))
		elif v < 2*u:
			image = cv2.warpAffine(image, leftShift, (rows, cols))
		return image

	def _VerticalShift(self, image, u=0.33, v=1.0):
		downShift = np.float32([[1,0,0],[0,1,15]])
		upShift = np.float32([[1,0,0],[0,1,-15]])
		rows,cols = image.shape[:2]
		if v < u:
			image = cv2.warpAffine(image, downShift, (rows, cols))
		elif v < 2*u:
			image = cv2.warpAffine(image, upShift, (rows, cols))
		return image

	def Shift(self, image):
		image = self._HorizontalShift(image, u=0.33, v=np.random.random())
		image = self._VerticalShift(image, u=0.33, v=np.random.random())
		return image

	def Rotate(self, image):
		u = 0.125
		v = np.random.random()
		rows,cols = image.shape[:2]
		angle = int(v/u)
		RotationMatrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle*45, 1)
		image = cv2.warpAffine(image, RotationMatrix, (rows,cols))
		return image

	def _zoomin(self, image, u=0.2, v=1.0):
		rows, cols = image.shape[:2]
		image = cv2.resize(image, (int(rows*1.2),int(cols*1.2)))
		if v < u:
			image = image[:rows, :cols]
		elif v < 2*u:
			image = image[(int(rows*1.2)-rows):, (int(cols*1.2)-cols):]
		elif v < 3*u:
			image = image[:rows, (int(cols*1.2)-cols):]
		elif v < 4*u:
			image = image[(int(cols*1.2)-rows):, :cols]
		else:
			image = image[int(rows*1.2/2)-int(rows/2):int(rows*1.2/2)+int(rows/2+1), 
				      int(rows*1.2/2)-int(cols/2):int(rows*1.2/2)+int(cols/2+1)]
		return image

	def _zoomout(self, image, u=0.2, v=1.0):
		rows, cols, ch = image.shape[:3]
		imagezero = np.zeros( (rows, cols, ch) ,dtype=np.uint8)
		image = cv2.resize(image, (int(rows*0.8),int(cols*0.8)))
		if v < u:
			imagezero[:len(image[0]), :len(image[1])] = image
		if v < 2*u:
			imagezero[:len(image[0]), cols-len(image[1]):] = image
		if v < 3*u:
			imagezero[rows-len(image[0]):, :len(image[1])] = image
		if v < 4*u:
			imagezero[rows-len(image[0]):, cols-len(image[1]):] = image
		else:
			imagezero[int(rows/2)-int(len(image[0])/2):int(rows/2)+int(len(image[0])/2), 
				  int(cols/2)-int(len(image[1])/2):int(cols/2)+int(len(image[1])/2)] = image
		return imagezero

	def Zoom(self, image, u=0.33):
		v = np.random.random()
		if v < u:
			image = self._zoomin(image, u=0.2, v=np.random.random())
		elif v < 2*u:
			image = self._zoomout(image, u=0.2, v=np.random.random())
		return image

	def Augment(self, image):
		if dataAugmentSettings.FLIP:
			image = self.Flip(image)
		if dataAugmentSettings.SHIFT:
			image = self.Shift(image)
		if dataAugmentSettings.ROTATE:
			image = self.Rotate(image)
		if dataAugmentSettings.ZOOM:
			image = self.Zoom(image)
		return image

if __name__ == '__main__':
	a = cv2.imread("Canvas1.jpg")
	Aug = DataAugmentation()
	a = Aug._zoomout(a)
	while True:
		cv2.imshow("a",a)
		cv2.waitKey(0)
		break