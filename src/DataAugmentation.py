import cv2
import numpy as np
import settings.DataAugmentSettings as dataAugmentSettings
import RadarImage

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
		probability = np.random.random()
		operations = ""
		if probability < PROBILITY_THRESHOLD_:
			image = self._zoomin(image)
			operations += " ZoomIn;"

		elif probability < 2*PROBILITY_THRESHOLD_:
			image = self._zoomout(image)
			operations += " ZoomOut;"

		return image, operations

	def Shift(self, image, PROBILITY_THRESHOLD_=0.5):
		probability = np.random.random()
		operations = ""
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
		tempImage = cv2.warpAffine(inputImage_, RotationMatrix, (rows,cols))
		operations = " Rotate:" + "{0:.2f}".format(angle) + ";"

		return tempImage, operations

	def _mergeTwoImages(self, foreground_, background_):
		'''
		       BACKGROUND_BOUND  Edge  FOREGROUND_BOUND
				 /       /       /
		    background	/       /       /  foreground
			       /       /       /
		'''
		FOREGROUND_BOUND = 1e-5
		BACKGROUND_BOUND = 1e-6
		foregroundMask = foreground_ > FOREGROUND_BOUND
		backgroundMask = foreground_ < BACKGROUND_BOUND

		edgeMask = np.logical_and( (foreground_ <= FOREGROUND_BOUND), (foreground_ >= BACKGROUND_BOUND) )
		alpha = 0.3
		edge = (alpha*foreground_ + (1.-alpha)*background_) * edgeMask
		#edge = (foreground_ + background_) * edgeMask

		result = foregroundMask*foreground_ + backgroundMask*background_ + edge
		return result

	def _horizontalFlip(self, image, PROBILITY_THRESHOLD_=0.5):
		return cv2.flip(image, 1)

	def _verticalFlip(self, image, PROBILITY_THRESHOLD_=0.5):
		return cv2.flip(image, 0)

	def _horizontalShift(self, image):
		step = np.random.uniform(low=-15., high=15.)
		
		shiftMatrix = np.float32( [[1, 0, step], [0, 1, 0]] )
		rows,cols = image.shape[:2]
		image = cv2.warpAffine(image, shiftMatrix, (rows, cols))

		return image

	def _verticalShift(self, image):
		step = np.random.uniform(low=-15., high=15.)
		shiftMatrix = np.float32( [[1, 0, 0],[0, 1, step]] )
		rows,cols = image.shape[:2]
		image = cv2.warpAffine(image, shiftMatrix, (rows, cols))

		return image

	def _zoomin(self, inputImage_):
		scale = 1.2
		inputShape = np.array(inputImage_.shape[:2])
		enlargedShape = (inputShape * scale).astype(np.int)
		enlargedImage = cv2.resize(inputImage_, tuple(enlargedShape))

		origine = np.array( [0, 0] )
		sizeDiff = np.array(enlargedImage.shape[:2]) - inputShape
		upperLeftPoint = origine + sizeDiff/2
		lowerRightPoint = upperLeftPoint + inputShape

		'''
		    Note: 1. OpenCV image coordinate start from the Top-Left point.
			  2. rows(y) first, than the column(x) dimension
		'''
		croppedImage = enlargedImage[upperLeftPoint[1]:lowerRightPoint[1], upperLeftPoint[0]:lowerRightPoint[0]]
		return croppedImage


	def _zoomout(self, inputImage_):
		scale = 0.8
		inputShape = np.array(inputImage_.shape[:2])
		shrinkedShape = (inputShape * scale).astype(np.int)
		shrinkedImage = cv2.resize(inputImage_, tuple(shrinkedShape) )

		result = np.zeros(inputImage_.shape)
		origine = np.array( [0, 0] )
		sizeDiff = np.array(result.shape[:2]) - shrinkedShape
		upperLeftPoint = origine + sizeDiff/2
		lowerRightPoint = upperLeftPoint + shrinkedShape

		result[upperLeftPoint[1]:lowerRightPoint[1], upperLeftPoint[0]:lowerRightPoint[0]] = shrinkedImage
		return result
