from random import shuffle
from src.RadarImage import *

class DataManager:
	def __init__(self, DATA_SET_FILE_PATH_, VALIDATION_RATIO_):
		self.isNewEpoch = True
		self.epoch = 0
		self.cursor = 0
		self.step = 1
		dataReader = StatoilDataReader()
		self.listOfRadarImages = dataReader.Read(DATA_SET_FILE_PATH_)
		shuffle(self.listOfRadarImages)
		NUMBER_OF_VALIDATION_DATA = int( VALIDATION_RATIO_*len(self.listOfRadarImages) )
		self.listOfValidationData = self.listOfRadarImages[ : NUMBER_OF_VALIDATION_DATA]
		self.listOfTrainingData = self.listOfRadarImages[NUMBER_OF_VALIDATION_DATA : ]

	def GetTrainingBatch(self, BATCH_SIZE_):
		arrayOfImages = np.zeros( (BATCH_SIZE_, DATA_WIDTH, DATA_HEIGHT, 2) )
		arrayOfAngles = np.zeros( (BATCH_SIZE_, 1) )
		arrayOfLabels = np.zeros( (BATCH_SIZE_, 2) )

		outputIndex = 0
		while outputIndex < BATCH_SIZE_:
			currentRadarImage = self.listOfTrainingData[self.cursor]
			arrayOfImages[outputIndex, :, :, :] = currentRadarImage.GetTotalImage(False)
			arrayOfLabels[outputIndex, :] = currentRadarImage.angle
			if currentRadarImage.hasAnswer:
				if currentRadarImage.isIceberg:
					arrayOfLabels[outputIndex, :] = np.array([0., 1.])
				else:
					arrayOfLabels[outputIndex, :] = np.array([1., 0.])
			outputIndex += 1
			self.cursor += 1
			self.step += 1
			self.isNewEpoch = False
			if self.cursor >= len(self.listOfTrainingData):
				shuffle(self.listOfTrainingData)
				self.cursor = 0
				self.epoch += 1
				self.isNewEpoch = True

		return arrayOfImages, arrayOfAngles, arrayOfLabels

	def GetValidationSet(self):
		NUMBER_OF_VALIDATION_DATA = len(self.listOfValidationData)
		arrayOfImages = np.zeros( (NUMBER_OF_VALIDATION_DATA, DATA_WIDTH, DATA_HEIGHT, 2) )
		arrayOfAngles = np.zeros( (NUMBER_OF_VALIDATION_DATA, 1) )
		arrayOfLabels = np.zeros( (NUMBER_OF_VALIDATION_DATA, 2) )

		for i in range(NUMBER_OF_VALIDATION_DATA):
			currentRadarImage = self.listOfTrainingData[i]
			arrayOfImages[i, :, :, :] = currentRadarImage.GetTotalImage(False)
			arrayOfLabels[i, :] = currentRadarImage.angle
			if currentRadarImage.hasAnswer:
				if currentRadarImage.isIceberg:
					arrayOfLabels[i, :] = np.array([0., 1.])
				else:
					arrayOfLabels[i, :] = np.array([1., 0.])

		return arrayOfImages, arrayOfAngles, arrayOfLabels

