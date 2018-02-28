import tensorflow as tf
import src.RadarImage as RadarImage
import settings.OutputSettings as outSettings
from settings.SubnetSettings import SubnetFactory

class IceNet:
	def __init__(self):
		self.isTraining = tf.placeholder(tf.bool)
		self.trainingStep = tf.placeholder(tf.int64)
		self.inputImage = tf.placeholder(tf.float32, 
						 [None, RadarImage.DATA_WIDTH, RadarImage.DATA_HEIGHT, RadarImage.DATA_CHANNELS])
		self.inputAngle = tf.placeholder(tf.float32, [None, 1])
		self.groundTruth = tf.placeholder(tf.float32, [None, outSettings.NUMBER_OF_CATEGORIES])

		self.softmax = None

		self.subnet = SubnetFactory(self.isTraining, self.trainingStep, self.inputImage, self.inputAngle, self.groundTruth)

	def Build(self):
		netOutput, updateSubnetOperation = self.subnet.Build()
		self.softmax = tf.nn.softmax(netOutput, name="tf.nn.softmax")
		with tf.name_scope("calculateLoss"):
			crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=netOutput, labels=self.groundTruth,
										name="tf.nn.softmax_cross_entropy_with_logits")
			loss = tf.reduce_mean(crossEntropy, name="tf.reduce_mean")
		accuracy = self.calculateAccuracy(netOutput, self.groundTruth)
		tf.summary.scalar('lossSummary', loss)
		tf.summary.scalar('accuracySummary', accuracy)

		return loss, accuracy, updateSubnetOperation

	def calculateAccuracy(self, netOutput_, groundTruth_):
		with tf.name_scope("calculateAccuracy"):
			correctPredictions = tf.equal(tf.argmax(netOutput_, 1), tf.argmax(groundTruth_, 1), name="tf.equal")
			correctPredictions = tf.reshape(correctPredictions, shape=[-1], name="tf.reshape")

			accuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32), name="tf.reduce_mean")
		return accuracy
