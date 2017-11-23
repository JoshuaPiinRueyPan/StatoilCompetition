import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.LayerHelper import *
import settings.OutputSettings as outSettings

class FullConvAlexnetTiny(SubnetBase):
	def __init__(self, isTraining_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_
		self.groundTruth = groundTruth_

	def Build(self):
		weights, biases = self.buildNetVariables()
		return self.buildNetBody(weights, biases)

	def buildNetVariables(self):
		weights = {
			'convW1': tf.Variable(tf.random_normal([3, 3, 2, 8], stddev=0.01)),
			'convW2': tf.Variable(tf.random_normal([3, 3, 8, 16], stddev=0.01)),
			'convW3': tf.Variable(tf.random_normal([3, 3, 16, 16], stddev=0.01)),
			'convW4': tf.Variable(tf.random_normal([3, 3, 16, 16], stddev=0.01)),
			'convW1x1' : tf.Variable(tf.random_normal([1, 1, 16, 2], stddev=0.01)),
		}

		biases = {
			'convb1': tf.Variable(tf.random_normal([8], stddev=0.01)),
			'convb2': tf.Variable(tf.random_normal([16], stddev=0.01)),
			'convb3': tf.Variable(tf.random_normal([16], stddev=0.01)),
			'convb4': tf.Variable(tf.random_normal([16], stddev=0.01)),
			'convb1x1' : tf.Variable(tf.random_normal([2], stddev=0.01)),
		}
		return weights, biases

	def buildNetBody(self, weights, biases):
		conv1 = convlayer('conv1', self.inputImage, weights['convW1'], biases['convb1'])
		pool1 = maxpool('pool1', conv1, k=2)
		norm1 = norm('norm1', pool1, lsize=4)

		conv2 = convlayer('conv2', norm1, weights['convW2'], biases['convb2'])
		pool2 = maxpool('pool2', conv2, k=2)
		norm2 = norm('norm3', pool2, lsize=4)

		conv3 = convlayer('conv3', norm2, weights['convW3'], biases['convb3'])
		pool3 = maxpool('pool3', conv3, k=2)
		norm3 = norm('norm3', pool3, lsize=4)

		conv4 = convlayer('conv4', norm3, weights['convW4'], biases['convb4'], padding='VALID')
		pool4 = maxpool('pool4', conv4, k=2)
		norm4 = norm('norm4', pool4, lsize=4)

		conv1x1 = convlayer('conv1x1', norm4, weights['convW1x1'], biases['convb1x1'], padding='VALID')
		maxPoolFinal = maxpool('poolFinal', conv1x1, k=5)
		output = tf.reshape(maxPoolFinal, shape=[-1, 2])
		
		return output

