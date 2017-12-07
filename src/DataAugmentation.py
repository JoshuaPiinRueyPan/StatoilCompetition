import cv2
import numpy as np
import settings.DataAugmentSettings as settings

class DataAugmentation:
	def Augment(self, inputImage_):
		performedOperations = ""
		tempImage = inputImage_
		# Flip
		tempImage, tempOp = self.Flip(tempImage)
		performedOperations += tempOp

		# Shift
		tempImage, tempOp = self.Shift(tempImage)
		performedOperations += tempOp

		# Rotate
		tempImage, tempOp = self.Rotate(tempImage)
		performedOperations += tempOp

		# Zoom
		tempImage, tempOp = self.Zoom(tempImage)
		performedOperations += tempOp

		mergedImage = self._mergeTwoImages(foreground_=tempImage, background_=inputImage_)

		return mergedImage, performedOperations

	def __init__(self):
		pass

	def Flip(self, image):
		operations = ""
		probability = np.random.random()
		if probability < settings.PROBABILITY_TO_FLIP_IMAGE:
			image = self._horizontalFlip(image)
			operations += " H-Flip;"

		probability = np.random.random()
		if probability < settings.PROBABILITY_TO_FLIP_IMAGE:
			image = self._verticalFlip(image)
			operations += " V-Flip;"

		return image, operations



	def Zoom(self, image):
		operations = ""
		probability = np.random.random()
		if probability < settings.PROBABILITY_TO_ZOOM_IMAGE:
			zoomInProbility = np.random.random()
			ZOOM_IN_THRESHOLD = 0.5
			if zoomInProbility < ZOOM_IN_THRESHOLD:
				image = self._zoomIn(image)
				operations += " ZoomIn;"

			else:
				image = self._zoomOut(image)
				operations += " ZoomOut;"

		return image, operations

	def Shift(self, image):
		operations = ""
		probability = np.random.random()
		if probability < settings.PROBABILITY_TO_SHIFT_IMAGE:
			image = self._horizontalShift(image)
			operations += " H-Shift;"

		probability = np.random.random()
		if probability < settings.PROBABILITY_TO_SHIFT_IMAGE:
			image = self._verticalShift(image)
			operations += " V-Shift;"

		return image, operations

	def Rotate(self, image):
		operations = ""
		probability = np.random.random()
		if probability < settings.PROBABILITY_TO_ROTATE_IMAGE:
			rows,cols = image.shape[:2]
			angle = np.random.uniform(low=0., high=360.0)
			RotationMatrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
			image = cv2.warpAffine(image, RotationMatrix, (rows,cols), flags=cv2.INTER_NEAREST)
			operations = " Rotate:" + "{0:.2f}".format(angle) + ";"

		return image, operations

	def _mergeTwoImages(self, foreground_, background_):
		THRESHOLD = 1e-6
		foregroundMask = foreground_ > THRESHOLD
		backgroundMask = 1 - foregroundMask
		result = foregroundMask*foreground_ + backgroundMask*background_
		return result


	def _horizontalFlip(self, image):
		return cv2.flip(image, 1)

	def _verticalFlip(self, image):
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
