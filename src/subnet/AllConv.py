import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.LayerHelper import *
import settings.OutputSettings as outSettings

class AllConv(SubnetBase):
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
			'convW3': tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.01)),
			'convW4': tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)),
			'convW5': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01)),
			'convW6': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01)),
			'convW7': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01)),
			'convW8': tf.Variable(tf.random_normal([1, 1, 512, 256], stddev=0.01)),
			'output': tf.Variable(tf.random_normal([256, outSettings.NUMBER_OF_CATEGORIES], stddev=0.01))
		}

		biases = {
			'convb1': tf.Variable(tf.random_normal([8], stddev=0.01)),
			'convb2': tf.Variable(tf.random_normal([16], stddev=0.01)),
			'convb3': tf.Variable(tf.random_normal([32], stddev=0.01)),
			'convb4': tf.Variable(tf.random_normal([64], stddev=0.01)),
			'convb5': tf.Variable(tf.random_normal([128], stddev=0.01)),
			'convb6': tf.Variable(tf.random_normal([256], stddev=0.01)),
			'convb7': tf.Variable(tf.random_normal([512], stddev=0.01)),
			'convb8': tf.Variable(tf.random_normal([256], stddev=0.01)),
			'output'  : tf.Variable(tf.random_normal([outSettings.NUMBER_OF_CATEGORIES], stddev=0.01)),
		}
		return weights, biases

	def buildNetBody(self, weights, biases):
		conv1 = convlayer('conv1', self.inputImage, weights['convW1'], biases['convb1'])
		norm1 = norm('norm1', conv1, lsize=4)

		conv2 = convlayer('conv2', norm1, weights['convW2'], biases['convb2'])
		norm2 = norm('norm3', conv2, lsize=4)

		conv3 = convlayer('conv3', norm2, weights['convW3'], biases['convb3'])
		pool3 = maxpool('pool3', conv3, k=2)
		norm3 = norm('norm3', pool3, lsize=4)

		conv4 = convlayer('conv4', norm3, weights['convW4'], biases['convb4'])
		pool4 = maxpool('pool4', conv4, k=2)
		norm4 = norm('norm4', pool4, lsize=4)

		conv5 = convlayer('conv5', norm4, weights['convW5'], biases['convb5'])
		pool5 = maxpool('pool5', conv5, k=2)
		norm5 = norm('norm5', pool5, lsize=4)

		conv6 = convlayer('conv6', norm5, weights['convW6'], biases['convb6'])
		pool6 = maxpool('pool6', conv6, k=2)
		norm6 = norm('norm6', pool6, lsize=4)

		conv7 = convlayer('conv7', norm6, weights['convW7'], biases['convb7'])
		norm7 = norm('norm7', conv7, lsize=4)

		conv8 = convlayer('conv8', norm7, weights['convW8'], biases['convb8'])

		avgpool = tf.nn.avg_pool(conv8, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='VALID', name='avg_pool')
		avgpool = tf.reshape(avgpool, shape=[-1, 256])

		output = tf.add(tf.matmul(avgpool, weights['output']), biases['output'])
		return output

