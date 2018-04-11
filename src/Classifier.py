import tensorflow as tf
import src.RadarImage as RadarImage
import settings.OutputSettings as outSettings
from settings.SubnetSettings import SubnetFactory
import settings.TrainSettings as trainSettings

class Classifier:
	def __init__(self):
		self.isTraining = tf.placeholder(tf.bool)
		self.trainingStep = tf.placeholder(tf.int64)
		self.inputImage = tf.placeholder(tf.float32, 
						 [None, RadarImage.DATA_WIDTH, RadarImage.DATA_HEIGHT, RadarImage.DATA_CHANNELS])
		self.inputAngle = tf.placeholder(tf.float32, [None, 1])
		self.groundTruth = tf.placeholder(tf.float32, [None, outSettings.NUMBER_OF_CATEGORIES])

		self.net = SubnetFactory(self.isTraining, self.trainingStep, self.inputImage, self.inputAngle, self.groundTruth)

	def Build(self):
		self.logits, updateSubnetOperation = self.net.Build()
		self.predictions = tf.nn.softmax(self.logits, name="tf.nn.softmax")
		with tf.name_scope("calculateLoss"):
			crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.groundTruth,
										name="tf.nn.softmax_cross_entropy_with_logits")
			loss = tf.reduce_mean(crossEntropy, name="tf.reduce_mean")
		accuracy = self.calculateAccuracy(self.predictions, self.groundTruth)

		return loss, accuracy, updateSubnetOperation


	def calculateAccuracy(self, predictions_, groundTruth_):
		with tf.name_scope("calculateAccuracy"):
			correctPredictions = tf.equal(tf.argmax(predictions_, 1), tf.argmax(groundTruth_, 1), name="tf.equal")
			accuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32), name="tf.reduce_mean")
			return accuracy

