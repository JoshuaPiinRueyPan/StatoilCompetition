import cv2
import numpy as np
import settings.DataAugmentSettings as dataAugmentSettings

class DataAugmentation:
	def Augment(self, image):
		if dataAugmentSettings.FLIP:
			image = self.Flip(image)
		if dataAugmentSettings.ZOOM:
			image = self.Zoom(image)
		if dataAugmentSettings.SHIFT:
			image = self.Shift(image)
		if dataAugmentSettings.ROTATE:
			image = self.Rotate(image)
		return image

	def __init__(self):
		pass

	def Flip(self, image, PROBILITY_THRESHOLD_=0.5):
		image = self._horizontalFlip(image, PROBILITY_THRESHOLD_)
		image = self._verticalFlip(image, PROBILITY_THRESHOLD_)
		return image



	def Zoom(self, image, PROBILITY_THRESHOLD_=0.33):
		randomValue = np.random.random()
		if randomValue < PROBILITY_THRESHOLD_:
			image = self._zoomin(image)

		elif randomValue < 2*PROBILITY_THRESHOLD_:
			image = self._zoomout(image)

		return image

	def Shift(self, image):
		image = self._horizontalShift(image, PROBILITY_THRESHOLD_=0.33)
		image = self._verticalShift(image, PROBILITY_THRESHOLD_=0.33)
		return image

	def Rotate(self, image):
		randomValue = np.random.random()
		rows,cols = image.shape[:2]
		angle = randomValue  * 360.0
		RotationMatrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
		image = cv2.warpAffine(image, RotationMatrix, (rows,cols))
		return image

	def _horizontalFlip(self, image, PROBILITY_THRESHOLD_=0.5):
		randomValue = np.random.random()
		if randomValue < PROBILITY_THRESHOLD_:
			image = cv2.flip(image, 1)
		return image

	def _verticalFlip(self, image, PROBILITY_THRESHOLD_=0.5):
		randomValue = np.random.random()
		if randomValue < PROBILITY_THRESHOLD_:
			image = cv2.flip(image, 0)
		return image

	def _horizontalShift(self, image, PROBILITY_THRESHOLD_=0.33):
		randomValue = np.random.random()
		rightShift = np.float32([[1,0,15],[0,1,0]])
		leftShift = np.float32([[1,0,-15],[0,1,0]])
		rows,cols = image.shape[:2]
		if randomValue < PROBILITY_THRESHOLD_:
			image = cv2.warpAffine(image, rightShift, (rows, cols))
		elif randomValue < 2*PROBILITY_THRESHOLD_:
			image = cv2.warpAffine(image, leftShift, (rows, cols))
		return image

	def _verticalShift(self, image, PROBILITY_THRESHOLD_=0.33):
		randomValue = np.random.random()
		downShift = np.float32([[1,0,0],[0,1,15]])
		upShift = np.float32([[1,0,0],[0,1,-15]])
		rows,cols = image.shape[:2]
		if randomValue < PROBILITY_THRESHOLD_:
			image = cv2.warpAffine(image, downShift, (rows, cols))
		elif randomValue < 2*PROBILITY_THRESHOLD_:
			image = cv2.warpAffine(image, upShift, (rows, cols))
		return image

	def _zoomin(self, image):
		rows, cols = image.shape[:2]
		image = cv2.resize(image, (int(rows*1.2),int(cols*1.2)))
		image = image[int(rows*1.2/2)-int(rows/2):int(rows*1.2/2)+int(rows/2+1), 
			      int(rows*1.2/2)-int(cols/2):int(rows*1.2/2)+int(cols/2+1)]
		image = cv2.resize(image, (rows,cols))
		return image

	def _zoomout(self, image):
		rows, cols, ch = image.shape[:3]
		imagezero = np.zeros( (rows, cols, ch) ,dtype=np.uint8)
		image = cv2.resize(image, (int(rows*0.8),int(cols*0.8)))
		imagezero[int(rows/2)-int(len(image[0])/2):int(rows/2)+int(len(image[0])/2), 
			  int(cols/2)-int(len(image[1])/2):int(cols/2)+int(len(image[1])/2)] = image
		imagezero = cv2.resize(imagezero, (rows,cols))
		return imagezero


if __name__ == '__main__':
	a = cv2.imread("Canvas1.jpg")
	Aug = DataAugmentation()
	a = Aug.Augment(a)
	while True:
		cv2.imshow("a",a)
		cv2.waitKey(0)
		break
