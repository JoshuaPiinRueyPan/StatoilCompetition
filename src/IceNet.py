import tensorflow as tf
import src.RadarImage as RadarImage
import settings.OutputSettings as outSettings
from settings.SubnetSettings import SubnetFactory
import settings.TrainSettings as trainSettings

class IceNet:
	def __init__(self):
		self.isTraining = tf.placeholder(tf.bool)
		self.trainingStep = tf.placeholder(tf.int64)
		self.inputImage = tf.placeholder(tf.float32, 
						 [None, RadarImage.DATA_WIDTH, RadarImage.DATA_HEIGHT, RadarImage.DATA_CHANNELS])
		self.inputAngle = tf.placeholder(tf.float32, [None, 1])
		self.groundTruth = tf.placeholder(tf.float32, [None, outSettings.NUMBER_OF_CATEGORIES])

		self.subnet = SubnetFactory(self.isTraining, self.trainingStep, self.inputImage, self.inputAngle, self.groundTruth)

	def Build(self):
		self.netOutput, updateSubnetOperation = self.subnet.Build()
		self.softmaxedOutput = tf.nn.softmax(self.netOutput, name="tf.nn.softmax")

		return updateSubnetOperation


	def GetTrainingResultTensor(self):
		with tf.name_scope("trainResults"):
			crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.netOutput, labels=self.groundTruth,
										name="CrossEntropy")
			lossOp = tf.reduce_mean(crossEntropy, name="tf.reduce_mean")
			
			correctPredictions = tf.equal(tf.argmax(self.softmaxedOutput, 1),
						      tf.argmax(self.groundTruth, 1),
						      name="tf.equal")
			correctPredictions = tf.reshape(correctPredictions, shape=[-1], name="tf.reshape")
			accuracyOp = tf.reduce_mean(tf.cast(correctPredictions, tf.float32), name="tf.reduce_mean")

		tf.summary.scalar('lossSummary', lossOp)
		tf.summary.scalar('accuracySummary', accuracyOp)

		return lossOp, accuracyOp


	def GetValidationResultTensor(self):
		with tf.name_scope("validationResults"):
			if trainSettings.DOES_CALCULATE_VALIDATION_SET_AT_ONCE:
				return  self.GetTrainingResultTensor()

			else:
				listOfSplitedInputs = tf.split( self.inputImage,
								num_or_size_splits=self.getSplitedValidationSize(),
								axis=0)
				listOfCrossEntropyTensors = []
				listOfPredictions = []
				for eachBatch in listOfSplitedInputs:
					currentCrossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.netOutput,
												      labels=self.groundTruth,
												      name="CrossEntropy")
					listOfCrossEntropyTensors.append(currentCrossEntropy)

					currentPredictions = tf.equal(tf.argmax(self.softmaxedOutput, 1),
								      tf.argmax(self.groundTruth, 1),
								      name="tf.equal")
					currentPredictions = tf.reshape(currentPredictions, shape=[-1], name="tf.reshape")
					listOfPredictions.append(currentPredictions)

				lossOp = tf.reduce_mean(listOfCrossEntropyTensors, name="tf.reduce_mean")
				accuracyOp = tf.reduce_mean( tf.cast(listOfPredictions, tf.float32), name="tf.reduce_mean")
		tf.summary.scalar('lossSummary', lossOp)
		tf.summary.scalar('accuracySummary', accuracyOp)

		return lossOp, accuracyOp


				

	def getSplitedValidationSize(self):
		quotient = int(trainSettings.NUMBER_OF_VALIDATION_DATA) / int(trainSettings.BATCH_SIZE)
		remainder = int(trainSettings.NUMBER_OF_VALIDATION_DATA) % int(trainSettings.BATCH_SIZE)
		listOfSplitedValidationBlockSize = [trainSettings.BATCH_SIZE] * quotient
		listOfSplitedValidationBlockSize.append(remainder)

		return listOfSplitedValidationBlockSize


