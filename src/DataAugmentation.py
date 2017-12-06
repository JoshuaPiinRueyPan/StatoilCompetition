import cv2
import numpy as np
import settings.DataAugmentSettings as dataAugmentSettings

class DataAugmentation:
	def Augment(self, inputImage_):
		performedOperations = ""
		tempImage = inputImage_
		if dataAugmentSettings.FLIP:
			tempImage, tempOp = self.Flip(tempImage)
			performedOperations += tempOp

		if dataAugmentSettings.ZOOM:
			tempImage, tempOp = self.Zoom(tempImage)
			performedOperations += tempOp

		if dataAugmentSettings.SHIFT:
			tempImage, tempOp = self.Shift(tempImage)
			performedOperations += tempOp

		if dataAugmentSettings.ROTATE:
			tempImage, tempOp = self.Rotate(tempImage)
			performedOperations += tempOp

		mergedImage = self._mergeTwoImages(foreground_=tempImage, background_=inputImage_)
		#mergedImage = tempImage

		return mergedImage, performedOperations

	def __init__(self):
		pass

	def Flip(self, image, PROBILITY_THRESHOLD_=0.5):
		operations = ""
		probability = np.random.random()
		if probability < PROBILITY_THRESHOLD_:
			image = self._horizontalFlip(image, PROBILITY_THRESHOLD_)
			operations += " H-Flip;"

		probability = np.random.random()
		if probability < PROBILITY_THRESHOLD_:
			image = self._verticalFlip(image, PROBILITY_THRESHOLD_)
			operations += " V-Flip;"

		return image, operations



	def Zoom(self, image, PROBILITY_THRESHOLD_=0.33):
		operations = ""
		probability = np.random.random()
		if probability < PROBILITY_THRESHOLD_:
			image = self._zoomIn(image)
			operations += " ZoomIn;"

		elif probability < 2*PROBILITY_THRESHOLD_:
			image = self._zoomOut(image)
			operations += " ZoomOut;"

		return image, operations

	def Shift(self, image, PROBILITY_THRESHOLD_=0.5):
		operations = ""
		probability = np.random.random()
		if probability < PROBILITY_THRESHOLD_:
			image = self._horizontalShift(image)
			operations += " H-Shift;"

		probability = np.random.random()
		if probability < PROBILITY_THRESHOLD_:
			image = self._verticalShift(image)
			operations += " V-Shift;"

		return image, operations

	def Rotate(self, inputImage_):
		rows,cols = inputImage_.shape[:2]
		angle = np.random.uniform(low=0., high=360.0)
		RotationMatrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
		tempImage = cv2.warpAffine(inputImage_, RotationMatrix, (rows,cols), flags=cv2.INTER_NEAREST)
		operations = " Rotate:" + "{0:.2f}".format(angle) + ";"

		return tempImage, operations

	def _mergeTwoImages(self, foreground_, background_):
		THRESHOLD = 1e-6
		foregroundMask = foreground_ > THRESHOLD
		backgroundMask = 1 - foregroundMask
		result = foregroundMask*foreground_ + backgroundMask*background_
		return result


	def _horizontalFlip(self, image, PROBILITY_THRESHOLD_=0.5):
		return cv2.flip(image, 1)

	def _verticalFlip(self, image, PROBILITY_THRESHOLD_=0.5):
		return cv2.flip(image, 0)

	def _horizontalShift(self, image):
		step = np.random.uniform(low=-15., high=15.)
		
		shiftMatrix = np.float32( [[1, 0, step], [0, 1, 0]] )
		image = cv2.warpAffine(image, shiftMatrix, image.shape[:2], flags=cv2.INTER_NEAREST)

		return image

	def _verticalShift(self, image):
		step = np.random.uniform(low=-15., high=15.)
		shiftMatrix = np.float32( [[1, 0, 0],[0, 1, step]] )
		image = cv2.warpAffine(image, shiftMatrix, image.shape[:2], flags=cv2.INTER_NEAREST)

		return image
	'''
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
	'''


	def _zoomIn(self, inputImage_):
		scale = 1.2
		inputShape = np.array(inputImage_.shape[:2])
		enlargedShape = (inputShape * scale).astype(np.int)
		enlargedImage = cv2.resize(inputImage_, tuple(enlargedShape), interpolation=cv2.INTER_NEAREST)

		origine = np.array( [0, 0] )
		sizeDiff = np.array(enlargedImage.shape[:2]) - inputShape
		upperLeftPoint = (origine + sizeDiff/2).astype(np.int)
		lowerRightPoint = upperLeftPoint + inputShape

		'''
		    Note: 1. OpenCV image coordinate start from the Top-Left point.
			  2. rows(y) first, than the column(x) dimension
		'''
		croppedImage = enlargedImage[upperLeftPoint[1]:lowerRightPoint[1], upperLeftPoint[0]:lowerRightPoint[0]]
		return croppedImage


	def _zoomOut(self, inputImage_):
		scale = 0.8
		inputShape = np.array(inputImage_.shape[:2])
		shrinkedShape = (inputShape * scale).astype(np.int)
		shrinkedImage = cv2.resize(inputImage_, tuple(shrinkedShape), interpolation=cv2.INTER_NEAREST )

		result = np.zeros(inputImage_.shape)
		origine = np.array( [0, 0] )
		sizeDiff = np.array(result.shape[:2]) - shrinkedShape
		upperLeftPoint = (origine + sizeDiff/2).astype(np.int)
		lowerRightPoint = upperLeftPoint + shrinkedShape

		result[upperLeftPoint[1]:lowerRightPoint[1], upperLeftPoint[0]:lowerRightPoint[0]] = shrinkedImage
		return result
